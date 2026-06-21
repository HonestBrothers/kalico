// XYZ (fully coupled CoreXYZ) kinematics stepper pulse time generation
//
// Copyright (C) 2026  Kalico contributors
//
// This file may be distributed under the terms of the GNU GPLv3 license.
//
// All three motors (A, B, C) contribute to all three toolhead axes using a
// symmetric, invertible coupling matrix:
//     A =  x + y + z
//     B =  x - y - z
//     C = -x + y - z
// The inverse (used for forward kinematics in the Python module) is:
//     x =  0.5 * (A + B)
//     y =  0.5 * (A + C)
//     z = -0.5 * (B + C)

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
    return c.x - c.y - c.z;
}

static double
xyz_stepper_c_calc_position(struct stepper_kinematics *sk, struct move *m
                            , double move_time)
{
    struct coord c = move_get_coord(m, move_time);
    return -c.x + c.y - c.z;
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
    sk->active_flags = AF_X | AF_Y | AF_Z;
    return sk;
}
