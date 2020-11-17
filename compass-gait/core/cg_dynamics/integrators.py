"""Integrate hybrid systems with event detection."""

import numpy as np
import jax.numpy as jnp
import scipy
import scipy.integrate

from absl import logging


def get_times(ts, tf, dt):
    assert ts < tf
    TOL = 1e-10
    times = [ts]
    tc = ts
    while tc < tf:
        tnext = tc + dt
        if np.abs(tnext - tf) <= TOL or tnext >= tf:
            times.append(np.minimum(tnext, tf))
            return jnp.array(times)
        times.append(tnext)
        tc = tnext
    assert False


DIR_NEGATIVE_TO_POSITIVE = 1.0
DIR_POSITIVE_TO_NEGATIVE = -1.0
DIR_EITHER = 0.0


def _proxy(f):
    return lambda *args, **kwargs: f(*args, **kwargs)


def hybrid_integration_v2(
        f, event, direction, guard, y0_discrete, y0_cts, t0, tf, switch_callback=None):

    assert direction == DIR_POSITIVE_TO_NEGATIVE, "limitation for now"

    if switch_callback is None:
        # no-op switch_callback
        switch_callback = lambda *args, **kwargs: None

    def make_scipy_event(event, direction, terminal):
        fn = _proxy(event)
        fn.direction = direction
        fn.terminal = terminal
        return fn

    ycur_discrete = y0_discrete
    ycur_cts = y0_cts
    tcur = t0

    GUARD_TOL = 1e-4
    PROGRESS_TOL = 1e-4

    method = "RK45"
    while True:

        cur_witness = event(tcur, ycur_cts)
        # print(f'cur_witness is {cur_witness}')

        if cur_witness >= 0 and cur_witness <= GUARD_TOL:
            logging.debug("cur_witness %f, tcur %f", cur_witness, tcur)
            # avoid getting stuck
            res = scipy.integrate.solve_ivp(f, (tcur, tf), ycur_cts,
                                            dense_output=True,
                                            events=[make_scipy_event(event, direction, terminal=False)],
                                            method=method)
            if res.status == -1:
                raise ValueError("integration failed: {}".format(res.message))
            elif res.status == 1:
                raise RuntimeError("non_terminal event should not trigger")
            assert res.status == 0

            t_events, = res.t_events
            t_events = np.sort(t_events)
            t_events = t_events[np.where(t_events > (tcur + PROGRESS_TOL))]
            
            
            # this means that neither foot hit the ground in this interval
            if len(t_events) == 0:
                # no event fired outside of the interval [tcur, tcur + PROGRESS_TOL],
                # which we are choosing to ignore in order to make progress
                return ycur_discrete, res.sol(tf)

            # otherwise, log the number of times a foot hit the ground 
            logging.debug("t_events %s", t_events)

            # take the first time a foot hit ground
            tcur = t_events[0]
            ycur_discrete_before = np.array(ycur_discrete)
            ycur_cts_before = res.sol(tcur)
            ycur_discrete, ycur_cts = guard(tcur, ycur_discrete, ycur_cts_before)
            switch_callback(tcur,
                    ycur_discrete_before, ycur_cts_before,
                    ycur_discrete, ycur_cts)

        else:
            # we are confident that we can make enough forward progress before
            # the next event
            res = scipy.integrate.solve_ivp(f, (tcur, tf), ycur_cts,
                                            dense_output=True,
                                            events=[make_scipy_event(event, direction, terminal=True)],
                                            method=method)

            # integration failed
            if res.status == -1:
                raise ValueError("integration failed: {}".format(res.message))

            # solver successfully reached end of timespan
            elif res.status == 0:

                # ycur_discrete is new discrete state
                # res.sol(tf) is the derivative evaluated at tf --> new continuous state
                return ycur_discrete, res.sol(tf)

            # a termination event occured -- means we took a step
            elif res.status == 1:
                t_events, = res.t_events
                t_events = np.sort(t_events)
                tcur = t_events[0]
                ycur_discrete_before = np.array(ycur_discrete)
                ycur_cts_before = res.sol(tcur)
                ycur_discrete, ycur_cts = guard(tcur, ycur_discrete, ycur_cts_before)
                switch_callback(tcur,
                        ycur_discrete_before, ycur_cts_before,
                        ycur_discrete, ycur_cts)

