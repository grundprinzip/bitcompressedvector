/* Copyright (c) 2011 Hasso Plattner Institute
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <iostream>
#include <iomanip>
#include <sys/time.h>

using namespace std;

class Timer
{
	friend std::ostream& operator<<(std::ostream& os, Timer& t);

private:
	bool running;
	timeval startTime, endTime;

public:
	// 'running' is initially false.  A timer needs to be explicitly started
	// using 'start' or 'restart'
	Timer() : running(false), startTime(), endTime() { }

	void start(const char* msg = 0);
	void stop(const char* msg = 0);
	double elapsed_time();
	double start_time();
	double end_time();
}; // class timer

//===========================================================================
// Return the total time that the timer has been in the "running"
// state since it was first "started" or last "restarted".  For
// "short" time periods (less than an hour), the actual cpu time
// used is reported instead of the elapsed time.

inline double Timer::elapsed_time()
{
	long seconds, useconds;

	seconds  = endTime.tv_sec  - startTime.tv_sec;
	useconds = endTime.tv_usec - startTime.tv_usec;

	return seconds + useconds / 1000.0 / 1000.0;
} // timer::elapsed_time

inline double Timer::start_time()
{
	long seconds, useconds;

	seconds  = startTime.tv_sec;
	useconds = startTime.tv_usec;

	return seconds + useconds / 1000.0 / 1000.0;
} // timer::elapsed_time

inline double Timer::end_time()
{
	long seconds, useconds;

	seconds  = endTime.tv_sec;
	useconds = endTime.tv_usec;

	return seconds + useconds / 1000.0 / 1000.0;
} // timer::elapsed_time

//===========================================================================
// Start a timer.  If it is already running, let it continue running.
// Print an optional message.

inline void Timer::start(const char* msg)
{
	// Print an optional message, something like "Starting timer t";
	if (msg) std::cout << msg << std::endl;

	// Return immediately if the timer is already running
	if (running) return;

	// Set timer status to running and set the start time
	running = true;
	gettimeofday(&startTime, NULL);

} // timer::start

//===========================================================================
// Stop the timer and print an optional message.

inline void Timer::stop(const char* msg)
{
	// Print an optional message, something like "Stopping timer t";
	if (msg) std::cout << msg << std::endl;

	// Compute accumulated running time and set timer status to not running
	if (running) gettimeofday(&endTime, NULL);
	running = false;

} // timer::stop

//===========================================================================
// Print out an optional message followed by the current timer timing.


//===========================================================================
// Allow timers to be printed to ostreams using the syntax 'os << t'
// for an ostream 'os' and a timer 't'.  For example, "cout << t" will
// print out the total amount of time 't' has been "running".

/*inline std::ostream& operator<<(std::ostream& os, timer& t)
{
	os //<< std::setprecision(8) << std::setiosflags(std::ios::fixed)
	<< t.acc_time + (t.running ? t.elapsed_time() : 0) ;
	os = os / 1000
			return os;
}
*/
//===========================================================================

#endif // TIMER_H
