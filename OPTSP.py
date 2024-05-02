import streamlit as st
import folium
import numpy as np
import pandas as pd
import itertools
from streamlit_folium import folium_static
from scipy.spatial.distance import pdist, squareform
from geopy.distance import geodesic
import random
import math
import time
import matplotlib.pyplot as plt

# Function to calculate the distance between two points using geopy
def calculate_distance(point1, point2):
    return geodesic(point1, point2).kilometers

# Function to add markers and draw a path on the map
def add_markers_and_path(map_object, points, path):
    for point in points:
        folium.Marker(
            location=[point[0], point[1]],
            popup=f'({point[0]}, {point[1]})'
        ).add_to(map_object)
    # Draw the path if available
    if path:
        folium.PolyLine(path, color='blue', weight=5, opacity=0.7).add_to(map_object)

# Function to solve the TSP using various algorithms
def nearest_neighbor_algorithm(points):
    if len(points) < 2:
        return points  # If there's less than 2 points, return early

    start_point = points[0]
    solution = [start_point]
    current_point = start_point  # Initialize current_point with the start point
    remaining_points = set(points[1:])  # Create a set of remaining points, excluding the start point

    while remaining_points:
        # Find the nearest remaining point to the current point
        nearest_point = min(remaining_points, key=lambda point: calculate_distance(current_point, point))

        # Remove the nearest point from remaining points and add it to the solution
        remaining_points.remove(nearest_point)
        solution.append(nearest_point)

        # Update current point
        current_point = nearest_point

    return solution + [start_point]  # Return to start and close the loop


def brute_force_algorithm(points):
    min_distance = float('inf')
    shortest_path = None
    for perm in itertools.permutations(points):
        total_distance = sum(calculate_distance(perm[i], perm[i + 1]) for i in range(len(perm) - 1))
        total_distance += calculate_distance(perm[-1], perm[0])  # Add distance from last point back to the start
        if total_distance < min_distance:
            min_distance = total_distance
            shortest_path = perm + (perm[0],)  # Add the first point to close the loop
    return shortest_path

def random_sampling_algorithm(points):
    min_distance = float('inf')
    shortest_path = None
    num_samples = min(1000, len(points)**2)  # Limit the number of samples
    for _ in range(num_samples):
        sample_path = random.sample(points, len(points))
        total_distance = sum(calculate_distance(sample_path[i], sample_path[i+1]) for i in range(len(sample_path)-1))
        total_distance += calculate_distance(sample_path[-1], sample_path[0])  # Add distance from last point back to the start
        if total_distance < min_distance:
            min_distance = total_distance
            shortest_path = sample_path + [sample_path[0]]  # Add the first point to close the loop
    return shortest_path

def genetic_algorithm(points, population_size=50, generations=100):
    def create_individual(points):
        return random.sample(points, len(points))
    def fitness(individual):
        return sum(calculate_distance(individual[i], individual[i+1]) for i in range(len(individual)-1))
    def crossover(parent1, parent2):
        crossover_point = random.randint(0, len(parent1)-1)
        child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        return child
    def mutate(individual):
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    population = [create_individual(points) for _ in range(population_size)]
    for _ in range(generations):
        parents = random.choices(population, k=2, weights=[fitness(individual) for individual in population])
        child = crossover(parents[0], parents[1])
        if random.random() < 0.1:  # Mutation probability
            child = mutate(child)
        worst_idx = max(range(population_size), key=lambda i: fitness(population[i]))
        population[worst_idx] = child
    return min(population, key=fitness) + [min(population, key=fitness)[0]]  # Close the loop

def simulated_annealing_algorithm(points):
    current_solution = nearest_neighbor_algorithm(points)[:-1]  # Start from Nearest Neighbor and remove duplicate start
    initial_temperature = 1000
    cooling_rate = 0.003
    current_cost = sum(calculate_distance(current_solution[i], current_solution[i+1]) for i in range(len(current_solution)-1))
    current_cost += calculate_distance(current_solution[-1], current_solution[0])  # Close loop

    temperature = initial_temperature
    while temperature > 1:
        new_solution = current_solution[:]
        idx1, idx2 = random.sample(range(len(new_solution)), 2)
        new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]
        new_cost = sum(calculate_distance(new_solution[i], new_solution[i+1]) for i in range(len(new_solution)-1))
        new_cost += calculate_distance(new_solution[-1], new_solution[0])  # Close loop
        if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / temperature):
            current_solution = new_solution
            current_cost = new_cost
        temperature *= 1 - cooling_rate
    return current_solution + [current_solution[0]]  # Close loop

# Streamlit app
def app():
    st.title('Optimal Delivery Route System Using TSP Algorithms')

    # Initialize or update the session state for points
    if 'points' not in st.session_state:
        st.session_state.points = []

    # Form for adding new markers
    with st.form("points_input_add"):
        lat = st.number_input('Latitude', value=36.7014631, format="%.4f")
        lon = st.number_input('Longitude', value=-118.755997, format="%.4f")
        submitted = st.form_submit_button('Add location')
        if submitted:
            new_point = (lat, lon)
            if new_point not in st.session_state.points:
                st.session_state.points.append(new_point)
                st.success(f"Marker added at ({lat}, {lon})")
            else:
                st.error(f"Coordinates ({lat}, {lon}) already exist, give other coordinates.")

    # Display map with current markers
    m = folium.Map(location=[36.7014631, -118.755997], zoom_start=8, tiles='OpenStreetMap')
    add_markers_and_path(m, st.session_state.points, [])
    folium_static(m)

    # Checkbox selection for TSP algorithms
    st.write("Select the TSP algorithms you want to use:")
    tsp_algorithms = {
        "Nearest Neighbor": nearest_neighbor_algorithm,
        "Brute Force": brute_force_algorithm,
        "Random Sampling": random_sampling_algorithm,
        "Genetic Algorithm": genetic_algorithm,
        "Simulated Annealing": simulated_annealing_algorithm
    }

    selected_algorithms = []
    select_all = st.checkbox("Select All")
    if select_all:
        selected_algorithms = list(tsp_algorithms.keys())
    else:
        selected_algorithms = [name for name in tsp_algorithms if st.checkbox(name)]

    # Refresh Button
    if st.button('Refresh Points'):
        st.session_state.points = []
        st.write("Points have been reset.")

    # Button to compute the optimized route
    if st.button('Generate Optimized Route'):
        if len(st.session_state.points) >= 2 and selected_algorithms:
            execution_times = {}
            routes = {}
            total_distances = {}

            for algorithm in selected_algorithms:
                start_time = time.time()
                route = tsp_algorithms[algorithm](st.session_state.points)
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                execution_times[algorithm] = execution_time
                routes[algorithm] = route

                total_distance_km = sum(calculate_distance(route[i], route[i + 1]) for i in range(len(route) - 1))
                total_distance_km += calculate_distance(route[-1], route[0])  # add distance from last point back to the start
                total_distances[algorithm] = total_distance_km * 0.621371  # Convert to miles

            # Results DataFrame
            results_df = pd.DataFrame({
                "Algorithm": selected_algorithms,
                "Total Distance (miles)": [total_distances[alg] for alg in selected_algorithms],
                "Execution Time (ms)": [execution_times[alg] for alg in selected_algorithms],
                "Route": [" -> ".join([f'({point[0]}, {point[1]})' for point in routes[alg]]) for alg in selected_algorithms]
            })

            # Display results
            st.write("Optimization Results:")
            st.dataframe(results_df)  # Display as a DataFrame widget

            # Route breakdown and distance matrix for all selected algorithms
            for algorithm in selected_algorithms:
                with st.expander(f"Details for {algorithm}", expanded=False):
                    # Route Breakdown
                    st.write(f"Route Breakdown for {algorithm}:")
                    route = tsp_algorithms[algorithm](st.session_state.points)
                    # Create route pairs
                    route_pairs = [(route[i], route[i+1]) for i in range(len(route)-1)] + [(route[-1], route[0])]
                    # Distance computation
                    distances_km = [calculate_distance(pair[0], pair[1]) for pair in route_pairs]
                    distances_miles = [distance_km * 0.621371 for distance_km in distances_km]

                    route_data = pd.DataFrame({
                        "From": [f"Point {i+1} ({pair[0][0]}, {pair[0][1]})" for i, pair in enumerate(route_pairs)],
                        "To": [f"Point {i+2 if i+2 <= len(route) else 1} ({pair[1][0]}, {pair[1][1]})" for i, pair in enumerate(route_pairs)],
                        "Distance (miles)": distances_miles
                    })
                    st.dataframe(route_data)

                    # Distance Matrix
                    st.write(f"Distance Matrix for {algorithm}:")
                    distance_matrix = np.zeros((len(route), len(route)))
                    for i in range(len(route)):
                        for j in range(len(route)):
                            distance_matrix[i, j] = calculate_distance(route[i], route[j])
                    distance_matrix_miles = distance_matrix * 0.621371
                    labels = [f"Point {i+1} ({route[i][0]}, {route[i][1]})" for i in range(len(route))]
                    st.dataframe(pd.DataFrame(distance_matrix_miles, columns=labels, index=labels))

            # Best Algorithm
            best_algorithm = results_df["Algorithm"][results_df["Total Distance (miles)"].idxmin()]
            st.write(f"Best Optimized Route Provided by {best_algorithm}:")
            m = folium.Map(location=[36.7014631, -118.755997], zoom_start=10, tiles='OpenStreetMap')
            add_markers_and_path(m, st.session_state.points, routes[best_algorithm])
            folium_static(m)

            # Visualization
            st.subheader("Execution Times of TSP Algorithms")
            fig, ax = plt.subplots()
            ax.bar(results_df["Algorithm"], results_df["Execution Time (ms)"])
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Distance Comparison Line Graph")
            fig2, ax2 = plt.subplots()
            ax2.plot(results_df["Algorithm"], results_df["Total Distance (miles)"], marker='o')
            ax2.set_xlabel('Algorithms')
            ax2.set_ylabel('Distances')
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        else:
            st.error("Please add at least two locations and select at least one algorithm.")

if __name__ == "__main__":
    app()
