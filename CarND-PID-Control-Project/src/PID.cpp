#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    PID::Kp = Kp;
    PID::Ki = Ki;
    PID::Kd = Kd;
//    intialized step
    step = 1;
    d_error = p_error = i_error = total_error =  0.0;
}

void PID::UpdateError(double cte) {
    double pre_cte = p_error;
    d_error = cte - pre_cte;
    p_error = cte;
    i_error += cte;
    
    if(step > 100){
        total_error += cte * cte;
    }
    step ++;
}


double PID::TotalError() {
    total_error = - Kp * p_error - Kd * d_error - Ki * i_error;
    return total_error;
}

