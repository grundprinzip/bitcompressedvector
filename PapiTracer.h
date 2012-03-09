#ifndef HELPER_PAPI_TRACER_H
#define HELPER_PAPI_TRACER_H

#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include <sys/time.h>

using namespace std;

#ifdef USE_PAPI_TRACE

#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <papi.h>
#include <pthread.h>
#include <stdexcept>

using namespace boost::interprocess;
#endif

struct PapiTracer
{

    typedef long long ll_t;
    typedef std::pair<ll_t, ll_t> result_t;

    static int start(std::string eventName = "PAPI_TOT_INS")
    {
      
        if (eventName.compare("NO_PAPI") == 0)
            return -1;

#ifdef USE_PAPI_TRACE


    //ovverride event settings
    eventName = std::string(getenv("PAPI_EVENT"));

	static bool initialized = false;

	if(!initialized)
	{
		named_mutex mtx(open_or_create, "PAPI_MTX_INIT");
		scoped_lock<named_mutex> lock(mtx);
		if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
			throw runtime_error("PAPI could not be initialized");

		initialized = true;
	}
        
        if (PAPI_thread_init(pthread_self) != PAPI_OK)
            throw runtime_error("PAPI could not initialize thread");

        int retval;
        
        // Create the event set
        int eventSet = PAPI_NULL;
        if ((retval = PAPI_create_eventset(&eventSet)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));

        // Add two events
        if ((retval = PAPI_add_event(eventSet, PAPI_TOT_CYC)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));

        // Add the configurable event
        int eventCode;
        if (PAPI_event_name_to_code((char*) eventName.c_str(), &eventCode) != PAPI_OK)
            throw runtime_error("Could not create event from name " + eventName);

        if ((retval = PAPI_add_event(eventSet, eventCode)) != PAPI_OK)
            throw runtime_error("Could not add events to event set, maybe conflicting");


        // Start the measurement
        if ((retval = PAPI_start(eventSet)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));

        return eventSet;
#else
        return clock();
#endif
    }

    static int reset(int eventSet)
    {
        if (eventSet == -1)
            return -1;

#ifdef USE_PAPI_TRACE
        ll_t tmp[2] = {0,0};
        int retval;
        if ((retval = PAPI_stop(eventSet, tmp)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));

        if ((retval = PAPI_start(eventSet)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));

        return eventSet;
#else
        return clock();
#endif
    }

    static result_t stop(int eventSet)
    {

        if (eventSet == -1)
        {
            result_t result;
            result.first = 0;
            result.second = 0;
            return result;
        }

#ifdef USE_PAPI_TRACE
        ll_t tmp[2] = {0,0};
        int retval;
        if ((retval = PAPI_stop(eventSet, tmp)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));

        if ((retval = PAPI_cleanup_eventset(eventSet)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));


        if ((retval = PAPI_destroy_eventset(&eventSet)) != PAPI_OK)
            throw runtime_error(PAPI_strerror(retval));


        result_t result;
        result.first = tmp[0];
        result.second = tmp[1];
        return result;
#else
        clock_t end = clock();
        result_t result;
        result.first = end - eventSet;
        result.second = 0;
        return result;
#endif
    }
};



#endif // HELPER_PAPI_TRACER_H
