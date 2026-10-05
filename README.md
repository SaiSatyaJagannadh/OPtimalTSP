<div align="center">

# 🗺️ OptimalTSP — Efficient Delivery Route Planner

### Drop pins on a map, pick an algorithm, and watch five TSP solvers race to the shortest route.

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-Try_it_now-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://optsppy-ja6ayynhktkfes4fdsz4iw.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-maps-77B829?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)

**👉 [Open the live app](https://optsppy-ja6ayynhktkfes4fdsz4iw.streamlit.app/)**

</div>

---

## ✨ Features

- 📍 **Interactive map** — add or remove delivery stops by latitude/longitude on a Folium map
- 🧮 **Five algorithms, side by side** — compare route length and runtime
- 🌍 **Real distances** — geodesic (great-circle) kilometres via `geopy`, not flat Euclidean
- 📊 **Charts** — runtime and distance comparisons with Matplotlib

## 🧠 Algorithms

| Algorithm | Type | Complexity | Best for |
|---|---|---|---|
| **Brute force** | Exact | O(n!) | ≤ 9 stops — the ground truth |
| **Nearest neighbour** | Greedy heuristic | O(n²) | Instant, decent answer |
| **Random sampling** | Stochastic | O(k·n) | Baseline to beat |
| **Genetic algorithm** | Metaheuristic | O(g·p·n) | Larger sets, near-optimal |
| **Simulated annealing** | Metaheuristic | O(i·n) | Escaping local minima |

## 🚀 Run locally

```bash
git clone https://github.com/SaiSatyaJagannadh/OPtimalTSP.git && cd OPtimalTSP
pip install -r requirements.txt
streamlit run OPTSP.py
```

📄 Full write-up: [`Optimal Delivery Route System Using TSP Algorithms.pdf`](./Optimal%20Delivery%20Route%20System%20Using%20TSP%20Algorithms.pdf)

---

<div align="center">

**Built by [Sai Satya Jagannadh Doddipatla (DJ)](https://saisatyajagannadh.github.io/PersonalPortfolio/)** · ⭐ Star the repo if it helped

</div>
