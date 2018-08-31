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
#include <iomanip>
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
    particles.push_back(Particle{i, dist_x(gen), dist_y(gen), dist_theta(gen), weight_init});
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // sample from the CTRV transition PDF p(x_k | x^i_{k-1})
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // push particles through the CTRV model
  for (auto& p : particles)
  {
    if (yaw_rate == 0)
    {
      p.x += velocity * delta_t * cos(p.theta) + dist_x(gen);
      p.y += velocity * delta_t * sin(p.theta) + dist_y(gen);
      p.theta += dist_theta(gen);
    }
    else
    {
      p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
      p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen);
      p.theta += yaw_rate * delta_t + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> landmarks, std::vector<LandmarkObs> &observations)
{
  // DATA ASSOCIATION using NEAREST NEIGHBOR
  // For each landmark measurement we need to determine the landmark in the map to which this measurement 
  // belongs to (i.e. from which landmark the measurement originated from).
  for (auto& obs : observations)
  {
    double min_dist = numeric_limits<double>::max(); // sensor range
    for (auto lm : landmarks)
    {
      double d = dist(lm.x, lm.y, obs.x, obs.y);
      if (d < min_dist)
      {
        min_dist = d;
        obs.id = lm.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
  for (auto& p : particles)  // Why not auto p? Because we need to change p. See: https://bit.ly/2oqJjQU
  {
    // TRANSFORMATION of LIDAR MEASUREMENTS to MAP coordinates
    // Car measures the landmark positions using LIDAR sensor, the landmark measurements are in CAR coordinate frame
    // The real landmark positions are given in the MAP coordinate frame
    // To calculate distances between the measurement of the landmarks and the landmarks themselves, the measurements
    // need to be converted from the CAR coordinates to MAP coordinates.

    // list of landmark observations transformed into the MAP frame
    vector<LandmarkObs> observations_transformed;
    for (auto o : observations)
    {
      double tx = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
      double ty = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
      observations_transformed.push_back(LandmarkObs{o.id, tx, ty});
    }
    
    // pick landmarks within sensor range of the particle
    vector<LandmarkObs> landmarks_within_range;
    for (auto lm : map_landmarks.landmark_list)
      if (dist(lm.x_f, lm.y_f, p.x, p.y) <= sensor_range)
        landmarks_within_range.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});

    dataAssociation(landmarks_within_range, observations_transformed);

    // Once each measurement has a landmark associated with it, we can compute 
    // the likelihood of the particle given the landmark observations.
    double particle_likelihood = 1.0;
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double Z = 1/(2*M_PI*std_x*std_y);
    for (auto ot : observations_transformed)
    { 
      // get the closest associated landmark
      LandmarkObs closest_landmark;
      for (auto lm : landmarks_within_range)
        if (ot.id == lm.id)
          closest_landmark = lm;

      particle_likelihood *= Z * exp(-0.5*( pow(ot.x - closest_landmark.x, 2)/pow(std_x, 2) + 
                                            pow(ot.y - closest_landmark.y, 2)/pow(std_y, 2) ));
    }
    p.weight = particle_likelihood;
  }
  
  for (unsigned int i = 0; i < particles.size(); ++i)
    weights[i] = particles[i].weight;
}

void ParticleFilter::resample()
{
  // Resampling with replacement with probability proportional to particle weight.
  default_random_engine gen;
  discrete_distribution<unsigned int> dist_particle(weights.begin(), weights.end());

  vector<Particle> particles_new;
  for (unsigned int i = 0; i < num_particles; ++i)
    particles_new.push_back(particles[dist_particle(gen)]);
  particles = particles_new;
  // fill(weights.begin(), weights.end(), 1.0);
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
