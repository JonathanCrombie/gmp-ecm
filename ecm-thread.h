/* Private thread-local state for independent libecm calculations. */
#ifndef ECM_THREAD_H
#define ECM_THREAD_H
#if defined(_MSC_VER)
#define ECM_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define ECM_THREAD_LOCAL __thread
#else
#define ECM_THREAD_LOCAL _Thread_local
#endif
#endif
