# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program
   
[//]: # (Image References)
[image1]: output_image/car.png

## Goals
In this project our goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. We will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

## Data

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Path Generation Implementation

The implementation of path generation can be found in file [main.cpp](https://github.com/wzding/Self_Driving_Car_Nanodegree/blob/master/CarND-Path-Planning-Project/src/main.cpp). I use [spline function](http://kluge.in-chemnitz.de/opensource/spline/) which fits a line to given x and y points in a fairly smooth function, to create smooth trajectories.

The first attemp is to make sure the car is stay in its lane and goes at a constant velocity without violating its acceleration and jerk. Using the simulator, I can obtain its previous path points. If the car is just starting out and there is no previous path points, I use the car's state. If they are not empty, I make sure it's tangent by using the last and the second last point in that previous path. I calculate the x, y, yaw and velocity based off the end values of the previous path so that the behavior planner starts from the end of the old path.

In Frenet Coordinates, "s" is the distance along a lane while "d" is the distance away from the center dividing line of the road. Using this property, I convert my x & y coordinates into frenet coordinates which makes it easier to calculate where I want the car to be on the road. The car starts at the middle lane (d = 1) and I keep this value unchanged since I want the car to stay at the same lane. The car's upcoming waypoints include 2 previous points and the locations of my car in 30, 60 and 90 meters (lines 353-355).
```
vector<double> next_wp0 = getXY(car_s+30,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
vector<double> next_wp1 = getXY(car_s+60,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
vector<double> next_wp2 = getXY(car_s+90,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
``` 
Next, we transform the Frenet Coordinates to this local car's coordinates so that the last point of the previous path is always at 0(lines 365-372).
```
for (int i=0;i<ptsx.size();i++)
{
    double shift_x=ptsx[i]-ref_x;
    double shift_y=ptsy[i]-ref_y;
    
    ptsx[i] = (shift_x*cos(0-ref_yaw)-shift_y*sin(0-ref_yaw));
    ptsy[i] = (shift_x*sin(0-ref_yaw)+shift_y*cos(0-ref_yaw));
}
```
We then need to fill the rest of our path planner with previous points. And we set 50 as the number of points. And we need to tranfer the coordinates back to global coordinates (lines 408-427).
```
for (int i=1;i<=50-previous_path_x.size(); i++){
    double N = (target_dist/(.02*ref_vel/2.24));
    double x_point = x_add_on + (target_x)/N;
    double y_point = s(x_point);
    
    x_add_on = x_point;
    
    double x_ref = x_point;
    double y_ref = y_point;
    
    x_point = (x_ref*cos(ref_yaw)-y_ref*sin(ref_yaw));
    y_point = (x_ref*sin(ref_yaw)+y_ref*cos(ref_yaw));
    
    x_point += ref_x;
    y_point += ref_y;
    
    next_x_vals.push_back(x_point);
    next_y_vals.push_back(y_point);
    
}
```
However, the car misses some opportunities to change lane if I keep the car on the middle lane. In order to make lane change safely and efficiently, I check how close the car in front of me is. If this distance is less than 30 meters I will check opportunities by looking in front of or behind my vehicle's "s" position in other lanes. If the closest vehicle in an adjancent lane is more than 30 meters away, I want to change to that lane. This iterates through all three lanes, and checks both in front of and behind my car for the closest vehicles. If there is no opportunies found in other lanes, I'll just stay in the current lane and deccelerate my car's speed (lines 254-312).

```
 // check cars in different lanes
bool ahead = false;
bool left = false;
bool right = false;
for(int i=0;i<sensor_fusion.size();i++){ 
    float d = sensor_fusion[i][6];
    // check car lane
    int car_lane = -1;
    if(d >0 && d<4){
      car_lane = 0; // left
    }else if(d>4 && d<8){
      car_lane = 1; // middle
    }else if(d>8 && d<12){
      car_lane = 2; // right
    }
    if(car_lane<0){
      continue;
    }
    // check car speed
    double vx = sensor_fusion[i][3];
    double vy = sensor_fusion[i][4];
    double check_speed = sqrt(vx*vx + vy*vy);
    // car on the road
    double check_car_s = sensor_fusion[i][5];
    // estimate car's s position
    check_car_s += ((double)prev_size * .02*check_speed);
    if(car_lane==lane){
      //car in our lane
      ahead = ahead | ((check_car_s>car_s) && ((check_car_s-car_s)<30)); 
    }else if(car_lane-lane == -1){
      // car in on the left - 30m in front and 30m on the back
      left = left | ((check_car_s>car_s-30) && ((check_car_s-car_s)<30));   
    }else if(car_lane-lane == 1){
      // car in on the right
      right = right | ((check_car_s>car_s-30) && ((check_car_s-car_s)<30)); 
    }
}  
// check whether my car is too close to any car in front
if(ahead){
  if(!left && lane > 0){
    // no car on left
    lane--;
  }else if(!right && lane != 2){
    // no car on right
    lane++;
  }else{
    // car on left and right
    ref_vel -= .225; 
  }
}
else{
  if(lane!=1){
      if((lane==0 && !right) || (lane==2 && !left)){
        lane =1;
      }
  }
  if(ref_vel < 49.5){
    ref_vel += .225;
  }
}
```

## Result Analysis

1. The car drives according to the speed limit. For each point, I compare the velocity (`ref_vel`) to my target vehicle's speed (49.5 MPH), and either accelerate or deccelerate based on where I am in comparison. For example, I set the starting speed as 0 MPH, and the acceleration is .225 M/H^2, which is equivalent to 5 m/s^2 (lines 206, 293-313). 

2. Max Acceleration and Jerk are not exceeded.
As mentioned above, the max acceleration is .225 M/H^2 which is less than 10 m/s^2. 

3. The car stays in its lane, except for the time between changing lanes.

4. The car is able to change lanes.

5. The car does not have collisions and is able to drive at least 4.32 miles without incident.

![alt text][image1]

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

### Simulator.
The Term3 Simulator which contains the Path Planning Project can be downloaded from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).

---

## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

[Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

