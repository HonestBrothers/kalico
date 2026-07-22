// Direct functional test of the compiled model-inverse FF seam in kin_shaper.c.
// Wraps a cartesian x-stepper with input_shaper + FF params and calls the real
// calc_position_cb on hand-built trapq moves.
//
// Covers both accel paths:
//   hst == 0 : legacy raw staircase accel (a = 2*half_accel) -- must stay
//              bit-for-bit what it was before smoothing was added.
//   hst >  0 : accel derived from a +/-hst central difference of velocity.
//              Two properties matter: (1) inside a move it must reproduce the
//              exact constant accel (v is linear there, so the difference is
//              exact), and (2) ACROSS a move boundary where accel steps, the
//              FF position must be CONTINUOUS -- that jump is the stepcompress
//              / "Timer too close" source this smoothing exists to remove.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "list.h"
#include "itersolve.h"
#include "trapq.h"

struct stepper_kinematics *cartesian_stepper_alloc(char axis);
struct stepper_kinematics *input_shaper_alloc(void);
int input_shaper_set_sk(struct stepper_kinematics *sk,
                        struct stepper_kinematics *orig);
int input_shaper_set_ff_params(struct stepper_kinematics *sk, char axis,
                               int en, double c1, double c2, double hst);

static const double P1 = 1.0e-3, P2 = 2.0e-6;

// Three chained moves, velocity continuous, acceleration stepping 0 -> 8000 -> 0.
// Linked circularly so a walk can never dereference an unlinked node.
static struct move mv[3];

static void
build_moves(void)
{
    memset(mv, 0, sizeof(mv));
    for (int i = 0; i < 3; i++) {
        mv[i].move_t = 0.02;
        mv[i].axes_r.x = 1.0;
    }
    mv[0].start_v = 50.0;  mv[0].half_accel = 0.0;     mv[0].start_pos.x = 0.0;
    mv[1].start_v = 50.0;  mv[1].half_accel = 4000.0;  mv[1].start_pos.x = 1.0;
    mv[2].start_v = 210.0; mv[2].half_accel = 0.0;     mv[2].start_pos.x = 3.6;

    mv[0].node.next = &mv[1].node; mv[1].node.prev = &mv[0].node;
    mv[1].node.next = &mv[2].node; mv[2].node.prev = &mv[1].node;
    mv[2].node.next = &mv[0].node; mv[0].node.prev = &mv[2].node;
}

static double
expect_raw(const struct move *m, double t)
{
    double dist = m->start_v * t + m->half_accel * t * t;
    double v = m->start_v + 2.0 * m->half_accel * t;
    double a = 2.0 * m->half_accel;
    return m->start_pos.x + m->axes_r.x * (dist + P1 * v + P2 * a);
}

int main(void)
{
    struct stepper_kinematics *cart = cartesian_stepper_alloc('x');
    struct stepper_kinematics *is = input_shaper_alloc();
    if (input_shaper_set_sk(is, cart) < 0) { printf("set_sk FAIL\n"); return 1; }
    build_moves();
    int ok = 1;

    // --- 1. hst = 0: legacy raw accel, unchanged behaviour ------------------
    input_shaper_set_ff_params(is, 'x', 1, P1, P2, 0.0);
    double maxerr = 0.0;
    for (double t = 0.0; t <= 0.0200001; t += 0.005) {
        double got = is->calc_position_cb(is, &mv[1], t);
        double err = fabs(got - expect_raw(&mv[1], t));
        if (err > maxerr) maxerr = err;
    }
    printf("1. legacy raw accel   maxerr=%.2e  %s\n",
           maxerr, maxerr < 1e-9 ? "PASS" : "FAIL");
    if (maxerr >= 1e-9) ok = 0;

    // Size of the discontinuity the raw path leaves at the mv[1]->mv[2]
    // boundary: accel steps by 8000, so the FF position jumps by P2*8000.
    double raw_end = is->calc_position_cb(is, &mv[1], mv[1].move_t);
    double raw_start = is->calc_position_cb(is, &mv[2], 0.0);
    double raw_jump = fabs(raw_end - raw_start);

    // --- 2. hst > 0: inside a move the smoothed accel must be EXACT ---------
    const double HST = 0.002;
    input_shaper_set_ff_params(is, 'x', 1, P1, P2, HST);
    // t well inside mv[1] so the +/-HST window never leaves the move: v is
    // linear there, so the central difference returns exactly 2*half_accel.
    double t_mid = 0.010;
    double got_mid = is->calc_position_cb(is, &mv[1], t_mid);
    double err_mid = fabs(got_mid - expect_raw(&mv[1], t_mid));
    printf("2. smoothed, mid-move err=%.2e  %s\n",
           err_mid, err_mid < 1e-9 ? "PASS" : "FAIL");
    if (err_mid >= 1e-9) ok = 0;

    // --- 3. hst > 0: FF position continuous ACROSS the accel step ----------
    double sm_end = is->calc_position_cb(is, &mv[1], mv[1].move_t);
    double sm_start = is->calc_position_cb(is, &mv[2], 0.0);
    double sm_jump = fabs(sm_end - sm_start);
    printf("3. boundary jump      raw=%.3e  smoothed=%.3e  %s\n",
           raw_jump, sm_jump, sm_jump < 1e-9 ? "PASS" : "FAIL");
    if (sm_jump >= 1e-9) ok = 0;
    if (!(raw_jump > 1e-3)) {
        printf("   (test weak: raw path showed no jump to remove)\n");
        ok = 0;
    }

    // --- 4. FF off -> plain axis position, no correction --------------------
    input_shaper_set_ff_params(is, 'x', 0, 0.0, 0.0, 0.0);
    double t0 = 0.01;
    double got0 = is->calc_position_cb(is, &mv[1], t0);
    double want0 = mv[1].start_pos.x
        + (mv[1].start_v * t0 + mv[1].half_accel * t0 * t0);
    double err0 = fabs(got0 - want0);
    printf("4. FF off             err=%.2e  %s\n",
           err0, err0 < 1e-9 ? "PASS" : "FAIL");
    if (err0 >= 1e-9) ok = 0;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
