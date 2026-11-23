#include <arpa/inet.h>
#include <netdb.h>

static int numeric_loopback_getnameinfo(const struct sockaddr *address, socklen_t length,
                                       char *host, socklen_t host_length,
                                       char *service, socklen_t service_length, int flags) {
    if (address && address->sa_family == AF_INET && length >= sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *ipv4 = (const struct sockaddr_in *)address;
        if ((ntohl(ipv4->sin_addr.s_addr) >> 24) == 127)
            flags |= NI_NUMERICHOST;
    } else if (address && address->sa_family == AF_INET6 && length >= sizeof(struct sockaddr_in6)) {
        const struct in6_addr *ipv6 = &((const struct sockaddr_in6 *)address)->sin6_addr;
        if (IN6_IS_ADDR_LOOPBACK(ipv6) || (IN6_IS_ADDR_V4MAPPED(ipv6) && ipv6->s6_addr[12] == 127))
            flags |= NI_NUMERICHOST;
    }
    return getnameinfo(address, length, host, host_length, service, service_length, flags);
}

__attribute__((used, section("__DATA,__interpose")))
static struct {
    const void *replacement;
    const void *original;
} interpose_getnameinfo = {numeric_loopback_getnameinfo, getnameinfo};
