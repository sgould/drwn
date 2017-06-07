/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnThreadPool.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              (based on code by David Breeden <breeden@cs.stanford.edu>
**              and Ian Goodfellow <ia3n@cs.stanford.edu>)
**
*****************************************************************************/

#pragma once
#include <queue>

#if defined(_WIN32)||defined(WIN32)||defined(__WIN32__)
#define DRWN_USE_WIN32THREADS
#else
#define DRWN_USE_PTHREADS
#endif

#ifdef DRWN_USE_PTHREADS
#include <pthread.h>
#endif

#ifdef DRWN_USE_WIN32THREADS
#include <windows.h>
#include <process.h>
#undef min
#undef max
#endif

using namespace std;

// drwnThreadJob --------------------------------------------------------------

class drwnThreadPool;

//! \brief Interface for a thread job functor.
//! \sa drwnThreadPool
//! \sa \ref drwnThreadPoolDoc

class drwnThreadJob {
    friend class drwnThreadPool;

 private:
    drwnThreadPool *_owner;
    unsigned _threadId;
    bool _bAquiredLock;

 public:
    drwnThreadJob();
    virtual ~drwnThreadJob();

    //! thread functor called by drwnThreadPool with the appropriate \b threadId
    virtual void operator()() = 0;

 protected:
    //! obtain the id for the thread running (or that ran) this job. The return
    //! value is guaranteed to be between 0 and \p numThreads. Useful for accessing
    //! a global resource without colliding with other jobs.
    inline unsigned threadId() { return _threadId; }
    
    void lock();    //!< acquire a lock (on all jobs in the same thread pool)
    void unlock();  //!< release the lock (on all jobs in the same thread pool)
};

// drwnThreadPool -------------------------------------------------------------

//! \brief Implements a pool of threads for running concurrent jobs.
//! \sa drwnThreadJob
//! \sa \ref drwnThreadPoolDoc

class drwnThreadPool {
    friend class drwnThreadJob;

 public:
    static unsigned MAX_THREADS; //!< maximum number of threads allowed

 public:
    //! create a thread pool with \b size (<= MAX_THREADS) threads
    drwnThreadPool(const unsigned size = MAX_THREADS);
    ~drwnThreadPool();

    //! prepare the thread pool to take jobs
    void start();

    //! add a job to the queue
    void addJob(drwnThreadJob* job);

    //! finish the jobs in the queue and stop
    void finish(bool bShowStatus = false);

    //! return the number of threads running
    unsigned numThreads() const { return _nThreads; }
    //! return the number of jobs remaining in the queue
    unsigned numJobsRemaining();

private:
    // main thread function
#ifdef DRWN_USE_PTHREADS
	static void *runJobs(void *argPtr);
#endif
#ifdef DRWN_USE_WIN32THREADS
	static unsigned __stdcall runJobs(void *argPtr);
#endif

#if defined(DRWN_USE_PTHREADS)||defined(DRWN_USE_WIN32THREADS)
    struct drwnThreadArgs {
        drwnThreadPool *owner;
        unsigned threadId;
        int jobsProcessed;
    };

    queue<drwnThreadJob *> _jobQ;  //!< job queue

    unsigned _nThreads;            //!< number of threads

#ifdef DRWN_USE_PTHREADS
    pthread_t *_threads;          //!< bank of threads
    drwnThreadArgs *_args;        //!< thread arguments

    pthread_mutex_t _mutex;       //!< mutex to schedule threads
    pthread_cond_t _cond;         //!< condition variable to wait for jobs
#endif
#ifdef DRWN_USE_WIN32THREADS
    HANDLE *_threads;             //!< bank of threads
    drwnThreadArgs *_args;        //!< thread arguments

    CRITICAL_SECTION _mutex;      //!< mutex to schedule threads
    CONDITION_VARIABLE _cond;     //!< condition event to wait for jobs
#endif

    bool _bQuit;                  //!< off button

    bool _bProfilerEnabled;       //!< used for restoring state of code profiler
#else
    unsigned _nThreads;
    void *_threads;
    void *_args;
#endif
};

