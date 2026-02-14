#if canImport(Darwin)
    import Darwin
    private typealias PlatformLock = os_unfair_lock
#elseif canImport(Glibc)
    import Glibc
    #if os(FreeBSD) || os(OpenBSD)
        private typealias PlatformLock = pthread_mutex_t?
    #else
        private typealias PlatformLock = pthread_mutex_t
    #endif
#else
    #error("Unsupported platform")
#endif

struct Locked<State> {
    private final class Buffer: ManagedBuffer<State, PlatformLock> {
        deinit { withUnsafeMutablePointerToElements { $0.destroy() } }
    }

    private let buffer: ManagedBuffer<State, PlatformLock>

    init(_ state: State) {
        buffer = Buffer.create(minimumCapacity: 1) { buffer in
            buffer.withUnsafeMutablePointerToElements { PlatformLockPointer.initialize($0) }
            return state
        }
    }

    func access<T>(_ block: (inout State) throws -> T) rethrows -> T {
        try buffer.withUnsafeMutablePointers { header, lock in
            lock.lock()
            defer { lock.unlock() }
            return try block(&header.pointee)
        }
    }
}

extension Locked: @unchecked Sendable where State: Sendable {}

private typealias PlatformLockPointer = UnsafeMutablePointer<PlatformLock>
private extension PlatformLockPointer {
    static func initialize(_ pointer: PlatformLockPointer) {
        #if canImport(Darwin)
            pointer.initialize(to: os_unfair_lock())
        #elseif canImport(Glibc)
            let result = pthread_mutex_init(pointer, nil)
            precondition(result == 0, "pthread_mutex_init failed")
        #else
            #error("Unsupported platform")
        #endif
    }

    func destroy() {
        #if canImport(Glibc)
            let result = pthread_mutex_destroy(self)
            precondition(result == 0, "pthread_mutex_destroy failed")
        #endif
        deinitialize(count: 1)
    }

    func lock() {
        #if canImport(Darwin)
            os_unfair_lock_lock(self)
        #elseif canImport(Glibc)
            pthread_mutex_lock(self)
        #else
            #error("Unsupported platform")
        #endif
    }

    func unlock() {
        #if canImport(Darwin)
            os_unfair_lock_unlock(self)
        #elseif canImport(Glibc)
            let result = pthread_mutex_unlock(self)
            precondition(result == 0, "pthread_mutex_unlock failed")
        #else
            #error("Unsupported platform")
        #endif
    }
}
