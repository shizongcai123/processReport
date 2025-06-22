import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import time
from target_functions import *

import matplotlib.pyplot as plt
import numpy as np

class Particle:
    def __init__(self, dim):
        self.position = np.random.uniform(-100, 100, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

class MutationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MutationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x) # the delta porsition for a particle

def pso_with_dnn_mutation(objective_func, num_particles, dim, max_iter, use_dnn_ratio=0.5):
    start_time = time.time()  # Start timing
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')
    history = []

    # initilize DNN
    input_dim = dim * 4 # every particle have 4 type of states position,  velocity, best_position, global_best_position
    mutation_net = MutationNet(input_dim=input_dim, hidden_dim=64, output_dim=dim)
    optimizer = optim.Adam(mutation_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for iter in range(max_iter):
        X_train = []
        Y_train = []

        for particle in particles:
            score = objective_func(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        for i, particle in enumerate(particles):
            if np.random.rand() < use_dnn_ratio:
                state = np.concatenate([
                    particle.position,
                    particle.velocity,
                    particle.best_position,
                    global_best_position
                ])
                state_tensor = torch.tensor(state, dtype=torch.float32)
                delta = mutation_net(state_tensor).detach().numpy()
                new_position = particle.position + delta
                # print("position delta: ", delta)
            else:
                w = 0.7
                c1 = c2 = 1.5
                r1, r2 = np.random.rand(2)

                cognitive = c1 * r1 * (particle.best_position - particle.position)
                social = c2 * r2 * (global_best_position - particle.position)

                particle.velocity = w * particle.velocity + cognitive + social
                new_position = particle.position + particle.velocity

            X_train.append(np.concatenate([
                particle.position,
                particle.velocity,
                particle.best_position,
                global_best_position
            ]))
            Y_train.append(new_position - particle.position)
            particle.position = np.clip(new_position, -100, 100)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.float32)

        mutation_net.train()
        optimizer.zero_grad()
        output = mutation_net(X_train)
        loss = loss_fn(output, Y_train)
        loss.backward()
        optimizer.step()

        history.append(global_best_score)

    end_time = time.time()  # End timing
    computation_time = end_time - start_time

    return global_best_position, global_best_score, history, computation_time

def pso_with_random_mutation(objective_func, num_particles, dim, max_iter, use_random_ratio=0.5, mutation_scale=10):
    """PSO with random mutation for comparison with DNN version"""
    start_time = time.time()
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')
    history = []

    for iter in range(max_iter):
        # Evaluation phase
        for particle in particles:
            score = objective_func(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        # Update phase
        for particle in particles:
            if np.random.rand() < use_random_ratio:
                # Random mutation equivalent to DNN's delta
                delta = np.random.normal(0, mutation_scale, dim)
                new_position = particle.position + delta
                # print("random delta: ", delta)
            else:
                # Standard PSO update
                w = 0.7
                c1 = c2 = 1.5
                r1, r2 = np.random.rand(2)
                
                cognitive = c1 * r1 * (particle.best_position - particle.position)
                social = c2 * r2 * (global_best_position - particle.position)
                
                particle.velocity = w * particle.velocity + cognitive + social
                new_position = particle.position + particle.velocity

            particle.position = np.clip(new_position, -100, 100)

        history.append(global_best_score)

    end_time = time.time()
    computation_time = end_time - start_time
    return global_best_position, global_best_score, history, computation_time

def spso(objective_func, num_particles, dim, max_iter):
    start_time = time.time()
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')
    history = []

    for _ in range(max_iter):
        for particle in particles:
            score = objective_func(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        for particle in particles:
            w = 0.7
            c1 = c2 = 1.5
            r1, r2 = np.random.rand(2)

            cognitive_component = c1 * r1 * (particle.best_position - particle.position)
            social_component = c2 * r2 * (global_best_position - particle.position)

            particle.velocity = w * particle.velocity + cognitive_component + social_component
            particle.position = particle.position + particle.velocity
            particle.position = np.clip(particle.position, -100, 100)

        history.append(global_best_score)

    end_time = time.time()
    computation_time = end_time - start_time
    return global_best_position, global_best_score, history, computation_time

def ipsom(objective_func, num_particles, dim, max_iter, p_m=0.5, sigma=10):
    start_time = time.time()
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')
    history = []
    
    for _ in range(max_iter):
        for particle in particles:
            score = objective_func(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()
        
        for particle in particles:
            w = 0.7
            c1 = c2 = 1.5
            r1, r2 = np.random.rand(2)
            
            if np.random.rand() >= p_m:
                cognitive_component = c1 * r1 * (particle.best_position - particle.position)
                social_component = c2 * r2 * (global_best_position - particle.position)  
                particle.velocity = w * particle.velocity + cognitive_component + social_component                
                particle.position = particle.position + particle.velocity
            else:
                particle.position = particle.position + np.random.normal(0, sigma, dim)
            
            particle.position = np.clip(particle.position, -100, 100)
        
        history.append(global_best_score)
    
    end_time = time.time()
    computation_time = end_time - start_time
    return global_best_position, global_best_score, history, computation_time

def hpso(objective_func, num_particles, dim, max_iter, F=0.5, cr=0.9):
    start_time = time.time()  
    particles = [Particle(dim) for _ in range(num_particles)]  
    global_best_position = np.zeros(dim)  
    global_best_score = float('inf')  
    history = []  
    
    for _ in range(max_iter):  
        for particle in particles:  
            score = objective_func(particle.position)  
            if score < particle.best_score:  
                particle.best_score = score  
                particle.best_position = particle.position.copy()  
            if score < global_best_score:  
                global_best_score = score  
                global_best_position = particle.position.copy()  
        
        for particle in particles:  
            w = 0.7  
            c1 = c2 = 1.5  
            r1, r2 = np.random.rand(2)  
            
            cognitive_component = c1 * r1 * (particle.best_position - particle.position)  
            social_component = c2 * r2 * (global_best_position - particle.position)  
            
            if np.random.rand() < cr:  
                idx1, idx2 = np.random.randint(0, num_particles, 2)  
                diff_vector = F * (particles[idx1].position - particles[idx2].position)  
                particle.velocity = w * particle.velocity + diff_vector + cognitive_component + social_component  
            else:  
                particle.velocity = w * particle.velocity + cognitive_component + social_component  
            
            particle.position = particle.position + particle.velocity  
            particle.position = np.clip(particle.position, -100, 100)  
        
        history.append(global_best_score)  
    
    end_time = time.time()
    computation_time = end_time - start_time
    return global_best_position, global_best_score, history, computation_time

def apso(objective_func, num_particles, dim, max_iter):
    start_time = time.time()
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')
    
    w_max, w_min = 0.9, 0.4
    c_1i, c_1f = 2.5, 0.5
    c_2i, c_2f = 0.5, 2.5
    
    history = []
    
    for iter in range(max_iter):
        for particle in particles:
            score = objective_func(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()
        
        w = w_max - (w_max - w_min) * (iter / max_iter)
        c1 = c_1i - (c_1i - c_1f) * (iter / max_iter)
        c2 = c_2i + (c_2f - c_2i) * (iter / max_iter)
        
        for particle in particles:
            r1, r2 = np.random.rand(2)
            cognitive_component = c1 * r1 * (particle.best_position - particle.position)
            social_component = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive_component + social_component
            particle.position = particle.position + particle.velocity
            particle.position = np.clip(particle.position, -100, 100)
        
        history.append(global_best_score)
    
    end_time = time.time()
    computation_time = end_time - start_time
    return global_best_position, global_best_score, history, computation_time

def run_comparison(dim, path):
    num_particles = 50
    dim = dim
    path = path
    max_iter = 5000
    num_runs = 5

    all_results = {
        'SPSO': {'scores': [], 'histories': [], 'times': []},
        'APSO': {'scores': [], 'histories': [], 'times': []},
        'IPSOM': {'scores': [], 'histories': [], 'times': []},
        'HPSO': {'scores': [], 'histories': [], 'times': []},
        'random_mutation': {'scores': [], 'histories': [], 'times': []},
        'dnn_mutation': {'scores': [], 'histories': [], 'times': []},
    }

    for i in range(num_runs):
        print(f"Running iteration {i+1}/{num_runs}")
        target_function = Sphere # set target function
        
        # Run each algorithm and collect results
        _, score_spso, hist_spso, time_spso = spso(target_function, num_particles, dim, max_iter)
        _, score_apso, hist_apso, time_apso = apso(target_function, num_particles, dim, max_iter)
        _, score_ipsom, hist_ipsom, time_ipsom = ipsom(target_function, num_particles, dim, max_iter)
        _, score_hpso, hist_hpso, time_hpso = hpso(target_function, num_particles, dim, max_iter)
        _, best_score2, history2, time_dnn = pso_with_dnn_mutation(
            objective_func=target_function,
            num_particles=num_particles,
            dim=dim,
            max_iter=max_iter,
            use_dnn_ratio=0.5
        )
        _, best_score3, history3, time_random = pso_with_random_mutation(
            objective_func=target_function,
            num_particles=num_particles,
            dim=dim,
            max_iter=max_iter,
            use_random_ratio = 0.5
        )

        # Store results
        all_results['SPSO']['scores'].append(score_spso)
        all_results['SPSO']['histories'].append(hist_spso)
        all_results['SPSO']['times'].append(time_spso)
        
        all_results['APSO']['scores'].append(score_apso)
        all_results['APSO']['histories'].append(hist_apso)
        all_results['APSO']['times'].append(time_apso)
        
        all_results['IPSOM']['scores'].append(score_ipsom)
        all_results['IPSOM']['histories'].append(hist_ipsom)
        all_results['IPSOM']['times'].append(time_ipsom)
        
        all_results['HPSO']['scores'].append(score_hpso)
        all_results['HPSO']['histories'].append(hist_hpso)
        all_results['HPSO']['times'].append(time_hpso)
        
        all_results['dnn_mutation']['scores'].append(best_score2)
        all_results['dnn_mutation']['histories'].append(history2)
        all_results['dnn_mutation']['times'].append(time_dnn)

        all_results['random_mutation']['scores'].append(best_score3)
        all_results['random_mutation']['histories'].append(history3)
        all_results['random_mutation']['times'].append(time_random)
    
    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)
    # Save statistics to a text file
    stats_file = os.path.join(path, f'statistics_dim_{dim}.txt')
    with open(stats_file, 'w') as f:
        for alg in all_results:
            scores = all_results[alg]['scores']
            times = all_results[alg]['times']
            f.write(f"\n{alg} Statistics:\n")
            f.write(f"Mean Score: {np.mean(scores):.2e}\n")
            f.write(f"Std Score: {np.std(scores):.2e}\n")
            f.write(f"Best Score: {np.min(scores):.2e}\n")
            f.write(f"Worst Score: {np.max(scores):.2e}\n")
            f.write(f"Mean Time: {np.mean(times):.2f} seconds\n")
            f.write(f"Total Time: {np.sum(times):.2f} seconds\n")
    
    # Also print to console
    for alg in all_results:
        scores = all_results[alg]['scores']
        times = all_results[alg]['times']
        print(f"\n{alg} Statistics:")
        print(f"Mean Score: {np.mean(scores):.2e}")
        print(f"Std Score: {np.std(scores):.2e}")
        print(f"Best Score: {np.min(scores):.2e}")
        print(f"Worst Score: {np.max(scores):.2e}")
        print(f"Mean Time: {np.mean(times):.2f} seconds")
        print(f"Total Time: {np.sum(times):.2f} seconds")

    # Plot and save convergence curves
    plt.figure(figsize=(10, 8))
    for alg in all_results:
        histories = np.array(all_results[alg]['histories'])
        mean_history = np.mean(histories, axis=0)
        plt.plot(mean_history, label=f"{alg}")
    
    # min_y = 0 
    # max_y = 100000
    # plt.ylim(bottom=min_y, top=max_y)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (log scale)')
    plt.title(f'Convergence Curves Comparison (Dim={dim}, Runs={num_runs})')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_file = os.path.join(path, f'convergence_curves_dim_{dim}.png')
    plt.savefig(plot_file)
    # plt.show()
    plt.close()
    
    print(f"\nResults saved to: {path}")
    print(f"Statistics saved to: {stats_file}")
    print(f"Plot saved to: {plot_file}")

if __name__ == "__main__":
    for i in [30, 50, 100, 200]:
        print("run dim = ", i)
        run_comparison(dim = i, path = ".\Sphere")