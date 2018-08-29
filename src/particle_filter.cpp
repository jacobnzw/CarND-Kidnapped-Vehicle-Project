/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  num_particles = 100;

  // init particle weights
  double weight_init = 1.0;
  weights = std::vector<double>(num_particles, weight_init);

  // random number generators to initialize particle position and rotation angle
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (unsigned int i = 0; i < num_particles; ++i)
  {
    Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), weight_init};
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // push particles through the CTRV model
  for (auto p : particles)
  {
    // TODO: zero yaw_rate case
    p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
    p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen);
    p.theta = p.theta + yaw_rate * delta_t + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  for (unsigned int i = 0; i < observations.size(); ++i)
  {
    double min_dist = 1000;
    int min_id;
    for (unsigned int j = 0; j < predicted.size(); ++j)
    {
      double d = dist(predicted[i].x, predicted[i].y, observations[i].x, observations[i].y);
      if (d < min_dist)
      {
        min_dist = d;
        min_id = predicted[i].id;
      }
    }
    observations[i].id = min_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  // TRANSFORMATION of LIDAR MEASUREMENTS to MAP coordinates
  // Car measures the landmark positions using LIDAR sensor, the landmark measurements are in CAR coordinate frame
  // The real landmark positions are given in the MAP coordinate frame
  // To calculate distances between the measurement of the landmarks and the landmarks themselves, the measurements
  // need to be converted from the CAR coordinates to MAP coordinates.
  
  x_m = particles.x + cos(particles.theta);
  
  // particle parametrizes the transformation
  // TODO: for each particle, I need to transform each observation into it's frame
  // that may be what the Particle::sense_x and Particle::sense_y fields are for.

  // DATA ASSOCIATION using NEAREST NEIGHBOR
  // This is important step for data association, where for each landmark measurement we need to determine 
  // the landmark in the map to which this measurement belongs to. 
  // That is, I obtain some numbers and ask myself: "did I just measure position of the tree or the corner 
  // of that building?" This is what data association solves. To each measurement it assigns a landmark from the map.
  dataAssociation(predicted, observations);

  // Once each measurement has a landmark associated with it, we can compute 
  // the likelihood of the particle given the landmark observations.
}

void ParticleFilter::resample()
{
  // Resampling with replacement with probability proportional to particle weight.
  default_random_engine gen;
  discrete_distribution<unsigned int> dist_particle(weights.begin(), weights.end());

  vector<Particle> particles_new;
  for (unsigned int i = 0; i < num_particles; ++i)
  {
    particles_new.push_back(particles[dist_particle(gen)]);
  }
  particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
