#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>

#define SYS_WRITE0 0x04
#define SYS_EXIT   0x18
#define SYS_EXIT_EXTENDED 0x20

#define ADP_STOPPED_APPLICATION_EXIT   0x20026
#define ADP_STOPPED_RUNTIME_ERROR      0x20023

extern char end;
extern char __HeapLimit;

static inline int semihost_call(int op, const void *arg)
{
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void *_sbrk(ptrdiff_t increment)
{
    static char *heap_end = &end;
    char *prev_heap_end = heap_end;

    if ((heap_end + increment) > &__HeapLimit) {
        errno = ENOMEM;
        return (void *)-1;
    }

    heap_end += increment;
    return prev_heap_end;
}

int _write(int file, const char *ptr, int len)
{
    char chunk[96];
    int written = 0;

    (void)file;

    while (written < len) {
        int chunk_len = len - written;
        if (chunk_len > ((int)sizeof(chunk) - 1)) {
            chunk_len = (int)sizeof(chunk) - 1;
        }

        for (int i = 0; i < chunk_len; ++i) {
            chunk[i] = ptr[written + i];
        }
        chunk[chunk_len] = '\0';
        semihost_call(SYS_WRITE0, chunk);
        written += chunk_len;
    }

    return len;
}

int _close(int file)
{
    (void)file;
    return -1;
}

int _fstat(int file, struct stat *st)
{
    (void)file;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file)
{
    (void)file;
    return 1;
}

int _lseek(int file, int ptr, int dir)
{
    (void)file;
    (void)ptr;
    (void)dir;
    return 0;
}

int _read(int file, char *ptr, int len)
{
    (void)file;
    (void)ptr;
    (void)len;
    return 0;
}

void _exit(int status)
{
    uint32_t exit_block[2] = {
        ADP_STOPPED_APPLICATION_EXIT,
        (uint32_t)status
    };

    semihost_call(SYS_EXIT_EXTENDED, exit_block);
    semihost_call(
        SYS_EXIT,
        (void *)(intptr_t)(status == 0 ? ADP_STOPPED_APPLICATION_EXIT : ADP_STOPPED_RUNTIME_ERROR));
    for (;;) {
    }
}

int _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

int _getpid(void)
{
    return 1;
}
