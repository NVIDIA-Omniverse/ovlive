// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// A minimal UDP DNS server on an ephemeral IPv4 loopback port. It answers A queries for one hostname with 127.0.0.1,
// other record types for that name with NODATA, and every other name with NXDOMAIN. A gRPC target
// "dns://127.0.0.1:<port>/<name>:<port>" resolves through it, so gRPC's c-ares resolver receives a real UDP reply
// without any change to the machine's resolver configuration.
//
// Include it before any header that includes windows.h, which otherwise brings in the legacy winsock.h.
#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class LoopbackDnsServer {
public:
    explicit LoopbackDnsServer(std::string hostname) : hostname_(std::move(hostname)) {}
    LoopbackDnsServer(const LoopbackDnsServer&) = delete;
    LoopbackDnsServer& operator=(const LoopbackDnsServer&) = delete;
    ~LoopbackDnsServer() {
        stop();
        if (socket_ != kInvalidSocket) {
#if defined(_WIN32)
            closesocket(socket_);
#else
            close(socket_);
#endif
        }
#if defined(_WIN32)
        if (winsockStarted_) WSACleanup();
#endif
    }

    // Binds the socket and starts answering; false when the socket cannot be bound.
    bool start() {
#if defined(_WIN32)
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) return false;
        winsockStarted_ = true;
#endif
        socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (socket_ == kInvalidSocket) return false;
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) return false;
        SocketLength size = sizeof(address);
        if (getsockname(socket_, reinterpret_cast<sockaddr*>(&address), &size) != 0) return false;
        port_ = ntohs(address.sin_port);
        running_ = true;
        thread_ = std::thread([this] { serve(); });
        return true;
    }

    // Stops answering and joins the serving thread, so the counters below are final: a reply's count is published
    // after its send, which the client may observe first.
    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    uint16_t port() const { return port_; }
    // A replies answered with the loopback address.
    uint32_t addressReplies() const { return addressReplies_; }
    // Malformed queries, socket failures and short sends.
    uint32_t errors() const { return errors_; }

private:
#if defined(_WIN32)
    using Socket = SOCKET;
    using SocketLength = int;
    static constexpr Socket kInvalidSocket = INVALID_SOCKET;
    bool winsockStarted_ = false;
#else
    using Socket = int;
    using SocketLength = socklen_t;
    static constexpr Socket kInvalidSocket = -1;
#endif
    std::string hostname_;
    Socket socket_ = kInvalidSocket;
    uint16_t port_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<uint32_t> addressReplies_{0};
    std::atomic<uint32_t> errors_{0};
    std::thread thread_;

    void serve() {
        while (running_) {
            fd_set readable;
            FD_ZERO(&readable);
            FD_SET(socket_, &readable);
            timeval timeout{0, 50000};
#if defined(_WIN32)
            const int selected = select(0, &readable, nullptr, nullptr, &timeout);
#else
            const int selected = select(socket_ + 1, &readable, nullptr, nullptr, &timeout);
#endif
            if (selected == 0) continue;
            if (selected < 0) {
                ++errors_;
                return;
            }
            uint8_t query[512]{};
            sockaddr_in client{};
            SocketLength clientSize = sizeof(client);
            const int count = static_cast<int>(recvfrom(socket_, reinterpret_cast<char*>(query), sizeof(query), 0,
                reinterpret_cast<sockaddr*>(&client), &clientSize));
            // Only standard queries (QR=0, OPCODE=0) carrying exactly one question are answered.
            if (count < 17 || (query[2] & 0xf8) != 0 || query[4] != 0 || query[5] != 1) {
                ++errors_;
                continue;
            }
            const auto size = static_cast<size_t>(count);
            size_t end = 12;
            std::string name;
            while (end < size && query[end] != 0) {
                const size_t labelSize = query[end++];
                if (labelSize > 63 || labelSize > size - end) break;
                if (!name.empty()) name += '.';
                name.append(reinterpret_cast<const char*>(query + end), labelSize);
                end += labelSize;
            }
            if (end + 5 > size || query[end] != 0) {
                ++errors_;
                continue;
            }
            ++end;
            for (char& c : name)
                if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            const bool knownName = name == hostname_;
            const bool answerAddress =
                knownName && query[end] == 0 && query[end + 1] == 1 && query[end + 2] == 0 && query[end + 3] == 1;
            end += 4;
            std::vector<uint8_t> reply(query, query + end);
            reply[2] = static_cast<uint8_t>(0x84 | (query[2] & 1)); // Response, authoritative, RD preserved.
            reply[3] = knownName ? 0 : 3; // NOERROR or NXDOMAIN.
            reply[6] = 0;
            reply[7] = answerAddress ? 1 : 0;
            reply[8] = reply[9] = reply[10] = reply[11] = 0;
            if (answerAddress) {
                // Name pointer to the question, type A, class IN, TTL 0, the four-byte IPv4 loopback address.
                const uint8_t record[]{0xc0, 0x0c, 0, 1, 0, 1, 0, 0, 0, 0, 0, 4, 127, 0, 0, 1};
                reply.insert(reply.end(), record, record + sizeof(record));
            }
            const int sent = static_cast<int>(sendto(socket_, reinterpret_cast<const char*>(reply.data()),
                static_cast<int>(reply.size()), 0, reinterpret_cast<sockaddr*>(&client), clientSize));
            if (sent != static_cast<int>(reply.size()))
                ++errors_;
            else if (answerAddress)
                ++addressReplies_;
        }
    }
};
