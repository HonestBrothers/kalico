// CoreXYZ kinematics stepper pulse time generation
//
// Copyright (C) 2026  Kalico contributors
//
// This file may be distributed under the terms of the GNU GPLv3 license.
//
// CoreXYZ uses four motors (A, B, C, D) that each move on all three
// toolhead axes.  Z is a common-mode of all four belts while X and Y are
// differential:
//     A =  x + y + z
//     B =  x - y + z
//     C = -x - y + z
//     D = -x + y + z
// The (over-determined) forward transform used by the Python module is:
//     x = (A + B - C - D) / 4
//     y = (A - B - C + D) / 4
//     z = (A + B + C + D) / 4

#include <stdlib.h> // malloc
#include <string.h> // memset
#include "compiler.h" // __visible
#include "itersolve.h" // struct stepper_kinematics
#include "trapq.h" // move_get_coord

static double
xyz_stepper_a_calc_position(struct stepper_kinematics *sk, struct move *m
                            , double move_time)
{
    struct coord c = move_get_coord(m, move_time);
    return c.x + c.y + c.z;
}

static double
xyz_stepper_b_calc_position(struct stepper_kinematics *sk, struct move *m
                            , double move_time)
{
    struct coord c = move_get_coord(m, move_time);
    return c.x - c.y + c.z;
}

static double
xyz_stepper_c_calc_position(struct stepper_kinematics *sk, struct move *m
                            , double move_time)
{
    struct coord c = move_get_coord(m, move_time);
    return -c.x - c.y + c.z;
}

static double
xyz_stepper_d_calc_position(struct stepper_kinematics *sk, struct move *m
                            , double move_time)
{
    struct coord c = move_get_coord(m, move_time);
    return -c.x + c.y + c.z;
}

struct stepper_kinematics * __visible
xyz_stepper_alloc(char type)
{
    struct stepper_kinematics *sk = malloc(sizeof(*sk));
    memset(sk, 0, sizeof(*sk));
    if (type == 'a')
        sk->calc_position_cb = xyz_stepper_a_calc_position;
    else if (type == 'b')
        sk->calc_position_cb = xyz_stepper_b_calc_position;
    else if (type == 'c')
        sk->calc_position_cb = xyz_stepper_c_calc_position;
    else if (type == 'd')
        sk->calc_position_cb = xyz_stepper_d_calc_position;
    sk->active_flags = AF_X | AF_Y | AF_Z;
    return sk;
}
