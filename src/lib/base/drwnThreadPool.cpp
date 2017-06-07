/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnThreadPool.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              David Breeden <breeden@cs.stanford.edu>
**              Ian Goodfellow <ia3n@cs.stanford.edu>
**
*****************************************************************************/

#include <iostream>
#include <limits>

#include "drwnLogger.h"
#include "drwnCodeProfiler.h"
#include "drwnConfigManager.h"
#include "drwnThreadPool.h"

// Static members
unsigned drwnThreadPool::MAX_THREADS = 4;

// drwnThreadJob --------------------------------------------------------------

drwnThreadJob::drwnThreadJob() : _owner(NULL), _threadId(0), _bAquiredLock(false)
{
    // do nothing
}

drwnThreadJob::~drwnThreadJob()
{
    // release lock if aquired (and forgot to unlock)
    if (_bAquiredLock) {
        DRWN_LOG_ERROR_ONCE("threads locked on destruction");
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_lock(&_owner->_mutex);
#endif
#ifdef DRWN_USE_WIN32THREADS
        LeaveCriticalSection(&_owner->_mutex);
#endif
	}
}

void drwnThreadJob::lock()
{
    if (_owner != NULL) {
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_lock(&_owner->_mutex);
#endif
#ifdef DRWN_USE_WIN32THREADS
        EnterCriticalSection(&_owner->_mutex);
#endif
        _bAquiredLock = true;
    }
}

void drwnThreadJob::unlock()
{
    if (_owner != NULL) {
        DRWN_ASSERT(_bAquiredLock);
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_unlock(&_owner->_mutex);
#endif
#ifdef DRWN_USE_WIN32THREADS
        LeaveCriticalSection(&_owner->_mutex);
#endif
        _bAquiredLock = false;
    }
}

// drwnThreadPool -------------------------------------------------------------

// Constructor
drwnThreadPool::drwnThreadPool(const unsigned size) :
    _nThreads(0), _threads(NULL), _args(NULL)
{
#ifdef DRWN_USE_PTHREADS
    _nThreads = (size > MAX_THREADS) ? MAX_THREADS : size;
    if (_nThreads > 0) {
        _threads = new pthread_t[_nThreads];
        _args = new drwnThreadArgs[_nThreads];
    }
    pthread_mutex_init(&_mutex, NULL);
    pthread_cond_init(&_cond, NULL);
    _bQuit = true;

    _bProfilerEnabled = drwnCodeProfiler::enabled;
#endif

#ifdef DRWN_USE_WIN32THREADS
    _nThreads = (size > MAX_THREADS) ? MAX_THREADS : size;
    if (_nThreads > 0) {
        _threads = new HANDLE[_nThreads];
        _args = new drwnThreadArgs[_nThreads];
    }

    InitializeConditionVariable(&_cond);
    InitializeCriticalSection(&_mutex);

    _bQuit = true;

    _bProfilerEnabled = drwnCodeProfiler::enabled;
#endif
}

// Destructor
drwnThreadPool::~drwnThreadPool()
{
#ifdef DRWN_USE_PTHREADS
    // tell threads to stop
    if (!_bQuit) {
        _bQuit = true;
        pthread_mutex_lock(&_mutex);
        pthread_cond_broadcast(&_cond);
        pthread_mutex_unlock(&_mutex);

        // wait for them to finish
        for (unsigned i = 0; i < _nThreads; i++) {
            pthread_join(_threads[i], NULL);
        }
    }

    pthread_mutex_destroy(&_mutex);
    pthread_cond_destroy(&_cond);
    if (_threads != NULL) delete[] _threads;
    if (_args != NULL) delete[] _args;
#endif

#ifdef DRWN_USE_WIN32THREADS
    // tell threads to stop
    if (!_bQuit) {
        _bQuit = true;
        WakeAllConditionVariable(&_cond);

        // wait for them to finish
        for (unsigned i = 0; i < _nThreads; i++) {
            WaitForSingleObject(_threads[i], INFINITE);
            CloseHandle(_threads[i]);
        }
    }

    if (_threads != NULL) delete[] _threads;
    if (_args != NULL) delete[] _args;
#endif
}

// Spawn threads to take jobs
void drwnThreadPool::start()
{
#if defined(DRWN_USE_PTHREADS)||defined(DRWN_USE_WIN32THREADS)
    _bQuit = false;

    // turn off code profiling
    _bProfilerEnabled = drwnCodeProfiler::enabled;
    if ((_nThreads > 1) && _bProfilerEnabled) {
        DRWN_LOG_WARNING_ONCE("turning off code profiler during threaded operations");
        drwnCodeProfiler::enabled = false;
    }

    // start threads
    for (unsigned i = 0; i < _nThreads; i++) {
        _args[i].owner = this;
        _args[i].threadId = i;
        _args[i].jobsProcessed = 0;
#ifdef DRWN_USE_PTHREADS
        pthread_create(&_threads[i], NULL, runJobs, (void *)&_args[i]);
#endif
#ifdef DRWN_USE_WIN32THREADS
        _threads[i] = (HANDLE)_beginthreadex(NULL, 0, drwnThreadPool::runJobs, (LPVOID)&_args[i], 0, NULL);
#endif
    }
#endif
}

// Add a new job to the job queue
void drwnThreadPool::addJob(drwnThreadJob *job)
{
#if defined(DRWN_USE_PTHREADS)
    if (_nThreads > 0) {
        // push job onto queue and tell threads about it
        pthread_mutex_lock(&_mutex);
        _jobQ.push(job);
        pthread_cond_broadcast(&_cond);
        pthread_mutex_unlock(&_mutex);
    } else {
        // just do it in the main thread
        (*job)();
    }
#elif defined(DRWN_USE_WIN32THREADS)
    if (_nThreads > 0) {
        // push job onto queue and tell threads about it
        EnterCriticalSection(&_mutex);
        _jobQ.push(job);
        LeaveCriticalSection(&_mutex);
        WakeAllConditionVariable(&_cond);
    } else {
        // just do it in the main thread
        (*job)();
    }
#else
	// threading disabled so just do it in the main thread
    (*job)();
#endif
}

// Wait until jobs are finished, then return
void drwnThreadPool::finish(bool bShowStatus)
{
#if defined(DRWN_USE_PTHREADS)||defined(DRWN_USE_WIN32THREADS)
    unsigned nJobsRemaining = std::numeric_limits<unsigned>::max();

    while (true) {
        // if queue is empty, stop looping
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_lock(&_mutex);
#endif
#ifdef DRWN_USE_WIN32THREADS
        EnterCriticalSection(&_mutex);
#endif
        if (_jobQ.empty()) {
            break;
        }

        // show status
        if (bShowStatus && (nJobsRemaining > _jobQ.size())) {
            nJobsRemaining = (unsigned)_jobQ.size();
            DRWN_LOG_STATUS("..." << nJobsRemaining << " remaining in the job queue ");
        }

        // if not, let the other threads get to it and try again
#ifdef DRWN_USE_PTHREADS
        pthread_mutex_unlock(&_mutex);
#endif
#ifdef DRWN_USE_WIN32THREADS
        LeaveCriticalSection(&_mutex);
#endif
	}

    // tell the threads to quit
    _bQuit = true;
#ifdef DRWN_USE_PTHREADS
    pthread_cond_broadcast(&_cond);
    pthread_mutex_unlock(&_mutex);
#endif
#ifdef DRWN_USE_WIN32THREADS
    LeaveCriticalSection(&_mutex);
    WakeAllConditionVariable(&_cond);
#endif

    // now wait for them to be done
    for (unsigned i = 0; i < _nThreads; i++) {
#ifdef DRWN_USE_PTHREADS
        pthread_join(_threads[i], NULL);
#endif
#ifdef DRWN_USE_WIN32THREADS
        WaitForSingleObject(_threads[i], INFINITE);
        CloseHandle(_threads[i]);
#endif
        DRWN_LOG_DEBUG("...thread " << i << " finished after processing "
            << _args[i].jobsProcessed << " jobs");
    }

    // re-enable code profiling
    drwnCodeProfiler::enabled = _bProfilerEnabled;
#endif
}

// thread main function
#ifdef DRWN_USE_PTHREADS
void *drwnThreadPool::runJobs(void *argPtr)
{
    drwnThreadArgs *args = (drwnThreadArgs *)argPtr;
    drwnThreadPool *pool = args->owner;

    // keep asking for jobs until quit flag
    while (!pool->_bQuit) {

        // wait for job
        pthread_mutex_lock(&pool->_mutex);
        while (pool->_jobQ.empty() && !pool->_bQuit) {
            pthread_cond_wait(&pool->_cond, &pool->_mutex);
        }

        if (pool->_jobQ.empty()) {
            pthread_mutex_unlock(&pool->_mutex);
            continue;
        }

        // take job off queue
        drwnThreadJob *job = pool->_jobQ.front();
        pool->_jobQ.pop();

        // unlock queue
        pthread_mutex_unlock(&pool->_mutex);

        // do job
        job->_owner = pool;
        job->_threadId = args->threadId;
        (*job)();
        job->_owner = NULL;
        args->jobsProcessed += 1;
    }

    return NULL;
}
#endif

#ifdef DRWN_USE_WIN32THREADS
unsigned __stdcall drwnThreadPool::runJobs(void *argPtr)
{
    drwnThreadArgs *args = (drwnThreadArgs *)argPtr;
    drwnThreadPool *pool = args->owner;

    // keep asking for jobs until quit flag
    while (!pool->_bQuit) {

        // wait for job
        EnterCriticalSection(&pool->_mutex);
        while (pool->_jobQ.empty() && !pool->_bQuit) {
            SleepConditionVariableCS(&pool->_cond, &pool->_mutex, INFINITE);
        }

        if (pool->_jobQ.empty()) {
            LeaveCriticalSection(&pool->_mutex);
            continue;
        }

        // take job off queue
        drwnThreadJob *job = pool->_jobQ.front();
        pool->_jobQ.pop();

        // unlock queue
        LeaveCriticalSection(&pool->_mutex);

        // do job
        job->_owner = pool;
        job->_threadId = args->threadId;
        (*job)();
        job->_owner = NULL;
        args->jobsProcessed += 1;
    }

    _endthreadex(0);
    return 0;
}
#endif

unsigned drwnThreadPool::numJobsRemaining()
{
#if defined(DRWN_USE_PTHREADS)
    pthread_mutex_lock(&_mutex);     // acquire lock on queue
    unsigned n = _jobQ.size();       // get queue size
    pthread_mutex_unlock(&_mutex);   // release lock
    return n;
#elif defined(DRWN_USE_WIN32THREADS)
    EnterCriticalSection(&_mutex);       // acquire lock on queue
    unsigned n = (unsigned)_jobQ.size(); // get queue size
    LeaveCriticalSection(&_mutex);       // release lock
    return n;
#else
    return 0;
#endif
}

// drwnThreadPoolConfig -----------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnThreadPool
//! \b threads :: maximum number of concurrent threads

class drwnThreadPoolConfig : public drwnConfigurableModule {
public:
    drwnThreadPoolConfig() : drwnConfigurableModule("drwnThreadPool") { }
    ~drwnThreadPoolConfig() { }

    void usage(ostream &os) const {
        os << "      threads       :: maximum number of concurrent threads (default: "
           << drwnThreadPool::MAX_THREADS << ")\n";
    }

    void setConfiguration(const char *name, const char *value) {
        // number of threads
        if (!strcmp(name, "threads")) {
            drwnThreadPool::MAX_THREADS = atoi(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnThreadPoolConfig gThreadPoolConfig;
