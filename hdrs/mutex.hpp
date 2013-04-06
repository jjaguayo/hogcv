#ifndef _mutex_hpp
#define _mutex_hpp

#include <pthread.h>

/// @file
/// @brief Simple mutex class

/// @brief Simple mutex class
class Mutex {
public:
        /// Default constructor
        Mutex() {
                pthread_mutex_init( &m_mutex, NULL );
        }
 
        /// Lock the mutex
        void lock() {
                pthread_mutex_lock( &m_mutex );
        }

        /// Unlock the mutex
        void unlock() {
                pthread_mutex_unlock( &m_mutex );
        }
private:
        pthread_mutex_t m_mutex;
};

#endif
