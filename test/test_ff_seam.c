// Direct functional test of the compiled model-inverse FF seam in kin_shaper.c.
// Wraps a cartesian x-stepper with input_shaper + FF params and calls the real
// calc_position_cb on a hand-built trapq move, comparing to y + p1*v + p2*a.
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "itersolve.h"
#include "trapq.h"

struct stepper_kinematics *cartesian_stepper_alloc(char axis);
struct stepper_kinematics *input_shaper_alloc(void);
int input_shaper_set_sk(struct stepper_kinematics *sk,
                        struct stepper_kinematics *orig);
int input_shaper_set_ff_params(struct stepper_kinematics *sk, char axis,
                               int en, double c1, double c2);

int main(void)
{
    struct stepper_kinematics *cart = cartesian_stepper_alloc('x');
    struct stepper_kinematics *is = input_shaper_alloc();
    if (input_shaper_set_sk(is, cart) < 0) { printf("set_sk FAIL\n"); return 1; }
    double p1 = 1.0e-3, p2 = 2.0e-6;
    input_shaper_set_ff_params(is, 'x', 1, p1, p2);

    struct move m;
    memset(&m, 0, sizeof(m));
    m.print_time = 0.0;
    m.move_t = 0.05;
    m.start_v = 50.0;
    m.half_accel = 4000.0;   // accel = 8000 mm/s^2
    m.start_pos.x = 10.0;
    m.axes_r.x = 1.0;

    double maxerr = 0.0;
    for (double t = 0.0; t <= 0.0500001; t += 0.01) {
        double got = is->calc_position_cb(is, &m, t);
        double dist = m.start_v * t + m.half_accel * t * t;
        double v = m.start_v + 2.0 * m.half_accel * t;
        double a = 2.0 * m.half_accel;
        double want = m.start_pos.x + m.axes_r.x * (dist + p1 * v + p2 * a);
        double err = fabs(got - want);
        if (err > maxerr) maxerr = err;
        printf("  t=%.3f  got=%.6f  want=%.6f  err=%.2e\n", t, got, want, err);
    }

    // FF off -> plain axis position (no correction).
    input_shaper_set_ff_params(is, 'x', 0, 0.0, 0.0);
    double t0 = 0.02;
    double got0 = is->calc_position_cb(is, &m, t0);
    double want0 = m.start_pos.x + (m.start_v * t0 + m.half_accel * t0 * t0);
    double err0 = fabs(got0 - want0);
    printf("  FF off: got=%.6f want=%.6f err=%.2e\n", got0, want0, err0);

    int ok = (maxerr < 1e-9) && (err0 < 1e-9);
    printf("%s (maxerr=%.2e)\n", ok ? "PASS" : "FAIL", maxerr);
    return ok ? 0 : 1;
}
