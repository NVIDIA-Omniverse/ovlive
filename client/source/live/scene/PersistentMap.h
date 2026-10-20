// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneWork.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace omni::usd::live {
// An ordered map whose copies share structure: an AVL tree of reference-
// counted nodes, each holding its value through its own reference. A copy
// costs constant time. A mutation copies only the nodes on its search path
// that another copy also holds, and an edited value only when another node
// also holds it, so each edit copies a logarithmic number of nodes and one
// value, never changes what another copy holds, and copies no value but the
// edited one. A node or value that only this map holds is changed in place. Like a std::map, one map
// object is never mutated while another thread reads that object; distinct
// copies may be read and mutated on different threads.
//
// Iteration visits keys in ascending order. Every node an iteration visits
// and every node a mutation copies counts toward the scene work meter, and
// every key comparison toward its comparison count.
//
// A mutation that throws, such as on an allocation failure, leaves the map's
// content as it was: edit and insert allocate before they change anything,
// and erase works on a copy of its path that replaces the map's only at the
// end.
template<class K, class V, class Less = std::less<K>> class PersistentMap {
    struct Node {
        Node(K k, V v) : key(std::move(k)), value(std::make_shared<V>(std::move(v))) {}
        Node(const Node&) = default;
        K key;
        std::shared_ptr<V> value;
        std::shared_ptr<Node> left, right;
        int height = 1;
    };
    using Ptr = std::shared_ptr<Node>;

public:
    // In-order traversal over borrowed nodes; valid while the map it came from
    // is neither mutated nor destroyed.
    class const_iterator {
    public:
        using value_type = std::pair<const K&, const V&>;
        const_iterator() = default;
        value_type operator*() const { return {stack_.back()->key, *stack_.back()->value}; }
        const K& key() const { return stack_.back()->key; }
        const V& value() const { return *stack_.back()->value; }
        const_iterator& operator++() {
            const Node* node = stack_.back();
            stack_.pop_back();
            descend(node->right.get());
            return *this;
        }
        bool operator==(const const_iterator& other) const {
            return stack_.empty() ? other.stack_.empty() :
                                    !other.stack_.empty() && stack_.back() == other.stack_.back();
        }
        bool operator!=(const const_iterator& other) const { return !(*this == other); }

    private:
        friend class PersistentMap;
        void descend(const Node* node) {
            for (; node; node = node->left.get()) {
                sceneWork::count(1);
                stack_.push_back(node);
            }
        }
        std::vector<const Node*> stack_;
    };

    PersistentMap() = default;
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    void clear() noexcept {
        root_.reset();
        size_ = 0;
    }
    const_iterator begin() const {
        const_iterator result;
        result.descend(root_.get());
        return result;
    }
    const_iterator end() const { return {}; }
    // The first entry whose key is not less than `key`.
    const_iterator lower_bound(const K& key) const {
        const_iterator result;
        for (const Node* node = root_.get(); node;) {
            if (less(node->key, key)) node = node->right.get();
            else {
                result.stack_.push_back(node);
                node = node->left.get();
            }
        }
        return result;
    }
    const V* find(const K& key) const {
        // One comparison per level: the last node not less than `key` is the
        // only one that can hold it.
        const Node* candidate = nullptr;
        for (const Node* node = root_.get(); node;) {
            if (less(node->key, key)) node = node->right.get();
            else {
                candidate = node;
                node = node->left.get();
            }
        }
        return candidate && same(key, candidate->key) ? candidate->value.get() : nullptr;
    }
    size_t count(const K& key) const { return find(key) ? 1 : 0; }
    // The stored value of `key`, owned by this map alone, or null.
    V* edit(const K& key) {
        const auto found = route(key);
        if (!found.present) return nullptr;
        Node* node = own(follow(found));
        if (node->value.use_count() != 1) {
            sceneWork::count(1);
            node->value = std::make_shared<V>(*node->value);
        } else std::atomic_thread_fence(std::memory_order_acquire);
        return node->value.get();
    }
    // Stores `value` under an absent `key`; a present key keeps its value.
    bool insert(K key, V value) {
        const auto found = route(key);
        if (found.present) return false;
        insertAlong(found, std::move(key), std::move(value));
        return true;
    }
    // Stores `value` under `key`, replacing any value it holds.
    void assign(K key, V value) {
        const auto found = route(key);
        if (!found.present) {
            insertAlong(found, std::move(key), std::move(value));
            return;
        }
        sceneWork::count(1);
        auto stored = std::make_shared<V>(std::move(value));
        own(follow(found))->value = std::move(stored);
    }
    bool erase(const K& key) {
        const auto found = route(key);
        if (!found.present) return false;
        // Every node the erase changes is a copy while `root` shares the tree.
        Ptr root = root_;
        eraseAt(root, found, 0);
        root_.swap(root);
        --size_;
        return true;
    }
    // Calls `visit(key, before, after)` for every key that `before` and
    // `after` do not hold as one shared value, in ascending key order, with
    // null for the side that lacks it. Subtrees the two maps share are skipped
    // whole, so comparing a map with an edited copy of itself visits the
    // copied paths only. A key both sides hold as distinct values is reported
    // even when the values are equal.
    template<class Visit>
    static void difference(const PersistentMap& before, const PersistentMap& after, Visit&& visit) {
        struct Frame {
            const Node* node;
            bool entry; // The node's own entry, its left subtree done; otherwise its whole subtree.
        };
        std::vector<Frame> a, b;
        if (before.root_) a.push_back({before.root_.get(), false});
        if (after.root_) b.push_back({after.root_.get(), false});
        const auto expand = [](std::vector<Frame>& frames) {
            const Node* node = frames.back().node;
            frames.pop_back();
            sceneWork::count(1);
            if (node->right) frames.push_back({node->right.get(), false});
            frames.push_back({node, true});
            if (node->left) frames.push_back({node->left.get(), false});
        };
        while (!a.empty() || !b.empty()) {
            if (a.empty() || b.empty()) {
                const bool added = a.empty(); // Only `after` holds the rest.
                auto& rest = added ? b : a;
                if (!rest.back().entry) {
                    expand(rest);
                    continue;
                }
                const Node* node = rest.back().node;
                rest.pop_back();
                if (added) visit(node->key, static_cast<const V*>(nullptr), node->value.get());
                else visit(node->key, node->value.get(), static_cast<const V*>(nullptr));
                continue;
            }
            const auto x = a.back(), y = b.back();
            if (x.node == y.node && x.entry == y.entry) {
                a.pop_back();
                b.pop_back();
                continue;
            }
            if (!x.entry || !y.entry) {
                // Expand the taller subtree so a subtree both hold surfaces on
                // both sides at once; equal heights expand both.
                const int hx = x.entry ? 0 : x.node->height, hy = y.entry ? 0 : y.node->height;
                if (!x.entry && hx >= hy) expand(a);
                if (!y.entry && hy >= hx) expand(b);
                continue;
            }
            if (less(x.node->key, y.node->key)) {
                a.pop_back();
                visit(x.node->key, x.node->value.get(), static_cast<const V*>(nullptr));
            } else if (less(y.node->key, x.node->key)) {
                b.pop_back();
                visit(y.node->key, static_cast<const V*>(nullptr), y.node->value.get());
            } else {
                a.pop_back();
                b.pop_back();
                if (x.node->value != y.node->value) visit(x.node->key, x.node->value.get(), y.node->value.get());
            }
        }
    }

private:
    static bool less(const K& a, const K& b) {
        sceneWork::compare();
        return Less{}(a, b);
    }
    // Whether `key` equals `stored`, a key not less than it. Keys ordered by
    // std::less compare equal with operator==, which for interned keys such as
    // SdfPath is constant time; any other order costs one more comparison.
    static bool same(const K& key, const K& stored) {
        if constexpr (std::is_same_v<Less, std::less<K>>) return key == stored;
        else return !less(key, stored);
    }
    // An AVL tree of 2^64 nodes is less than 93 levels high.
    static constexpr size_t kMaxHeight = 128;
    // The turns from the root to the node that holds a key, or to the empty
    // link where an absent key belongs: right[i] when the path leaves its
    // i-th node to the right. A mutation follows it without comparing keys.
    struct Route {
        std::array<bool, kMaxHeight> right;
        size_t depth = 0;
        bool present = false;
    };
    Route route(const K& key) const {
        Route result;
        const Node* candidate = nullptr;
        size_t depth = 0, at = 0;
        for (const Node* node = root_.get(); node; ++depth) {
            result.right[depth] = less(node->key, key);
            if (result.right[depth]) node = node->right.get();
            else {
                candidate = node;
                at = depth;
                node = node->left.get();
            }
        }
        result.present = candidate && same(key, candidate->key);
        result.depth = result.present ? at : depth;
        return result;
    }
    // The link of the node a present key's route ends at, owning every node
    // above it.
    Ptr& follow(const Route& route) {
        Ptr* link = &root_;
        for (size_t i = 0; i < route.depth; ++i) {
            Node* node = own(*link);
            link = route.right[i] ? &node->right : &node->left;
        }
        return *link;
    }
    void insertAlong(const Route& route, K key, V value) {
        sceneWork::count(1);
        auto leaf = std::make_shared<Node>(std::move(key), std::move(value));
        insertAt(root_, std::move(leaf), route, 0);
        ++size_;
    }
    static int height(const Ptr& node) noexcept { return node ? node->height : 0; }
    static void update(Node& node) noexcept { node.height = 1 + std::max(height(node.left), height(node.right)); }
    // The node `link` names, copied first when another link also holds it.
    static Node* own(Ptr& link) {
        if (link.use_count() != 1) {
            sceneWork::count(1);
            link = std::make_shared<Node>(*link);
        } else {
            // The last other holder released the node; see its writes.
            std::atomic_thread_fence(std::memory_order_acquire);
        }
        return link.get();
    }
    // `link` is owned; its left child becomes its parent.
    static void rotateRight(Ptr& link) {
        own(link->left);
        Ptr child = std::move(link->left);
        link->left = std::move(child->right);
        update(*link);
        child->right = std::move(link);
        update(*child);
        link = std::move(child);
    }
    static void rotateLeft(Ptr& link) {
        own(link->right);
        Ptr child = std::move(link->right);
        link->right = std::move(child->left);
        update(*link);
        child->left = std::move(link);
        update(*child);
        link = std::move(child);
    }
    // `link` is owned.
    static void rebalance(Ptr& link) {
        update(*link);
        const int balance = height(link->left) - height(link->right);
        if (balance > 1) {
            if (height(link->left->left) < height(link->left->right)) {
                own(link->left);
                rotateLeft(link->left);
            }
            rotateRight(link);
        } else if (balance < -1) {
            if (height(link->right->right) < height(link->right->left)) {
                own(link->right);
                rotateRight(link->right);
            }
            rotateLeft(link);
        }
    }
    // Owns each node on the leaf's path before linking it, so rebalancing,
    // whose rotations change only nodes on that path, never allocates.
    static void insertAt(Ptr& link, Ptr&& leaf, const Route& route, size_t depth) {
        if (!link) {
            link = std::move(leaf);
            return;
        }
        Node* node = own(link);
        insertAt(route.right[depth] ? node->right : node->left, std::move(leaf), route, depth + 1);
        rebalance(link);
    }
    // Detaches and returns the least node of the nonempty subtree `link`.
    static Ptr takeLeast(Ptr& link) {
        Node* node = own(link);
        if (!node->left) {
            Ptr least = std::move(link);
            link = std::move(least->right);
            return least;
        }
        Ptr least = takeLeast(node->left);
        rebalance(link);
        return least;
    }
    // The route's key is present in the subtree `link`, `depth` turns below
    // the root.
    static void eraseAt(Ptr& link, const Route& route, size_t depth) {
        Node* node = own(link);
        if (depth < route.depth) eraseAt(route.right[depth] ? node->right : node->left, route, depth + 1);
        else if (!node->left) {
            link = Ptr(std::move(node->right));
            return;
        } else if (!node->right) {
            link = Ptr(std::move(node->left));
            return;
        } else {
            Ptr least = takeLeast(node->right);
            least->left = std::move(node->left);
            least->right = std::move(node->right);
            link = std::move(least);
        }
        rebalance(link);
    }
    Ptr root_;
    size_t size_ = 0;
};
} // namespace omni::usd::live
