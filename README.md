# SCAME — Side-Channel Attack on Memristive Engines

![Project](https://img.shields.io/badge/research-side--channel%20attack-blue)

A research codebase implementing and evaluating side-channel attacks against memristive computing engines. This repository collects scripts, experiments, datasets, and notes used to study how memristor-based accelerators leak information through timing, power, or other side channels, and includes tools to reproduce core results.


## Highlights

- Reproducible experiments and scripts for simulating memristive engine workloads
- Data collection pipelines and analysis notebooks
- Attack implementations and defenses (where applicable)
- Clear instructions to run and reproduce results


## Motivation

Memristive engines are an emerging class of hardware accelerators with promising performance and energy characteristics. However, like many physical computing systems, they can leak sensitive information through side channels (timing, power, electromagnetic emanations, etc.). SCAME aims to provide a research toolkit to study these vulnerabilities, reproduce experiments, and explore mitigations.


## Quick Start

1. Clone the repository

   git clone https://github.com/Udaykirancheera15/SCAME.git
   
   cd SCAME

2. Create a virtual environment (recommended)

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows (PowerShell)

3. Install dependencies

   pip install -r requirements.txt

4. Run a sample experiment

   python experiments/run_demo.py

Note: If requirements.txt is not present, inspect the project files for instructions or run the notebooks in a Jupyter environment.


## Repository Structure (suggested)

- data/                — raw and processed datasets used in experiments
- notebooks/           — Jupyter notebooks for analysis and visualization
- experiments/         — scripts to run experiments and collect traces
- attacks/             — attack implementations (exploit code)
- defenses/            — mitigation experiments and prototypes
- utils/               — helper functions, data loaders, and plotting utilities
- results/             — output traces, models, and figures
- README.md            — this file


## Reproducing Results

To reproduce core experiments:

1. Ensure you have Python 3.8+ and the required dependencies installed.
2. Prepare datasets (if not included): follow instructions in data/README or use provided download scripts.
3. Run experiments from the experiments/ directory. For example:

   python experiments/collect_traces.py --config experiments/configs/demo.yaml
   python experiments/run_attack.py --traces results/demo/traces.pkl --config experiments/configs/attack_demo.yaml

4. Analyze results using notebooks:

   jupyter notebook notebooks/analysis.ipynb

If an experiment relies on specialized simulation code or external hardware, the corresponding README in experiments/ or a scripts/ folder contains details.


## Examples

- Demo trace collection and attack: experiments/run_demo.py
- Visualization of side-channel leakage: notebooks/leakage_viz.ipynb


## Best Practices & Ethics

Research into side-channel attacks can reveal vulnerabilities with real-world impact. Use this repository responsibly:

- Only run attacks on systems and devices you own or have explicit permission to test.
- Follow your institution's IRB / ethics guidelines and local laws.
- If you discover a vulnerability in third-party hardware, follow responsible disclosure practices.


## Contributing

Contributions are welcome. Please open an issue to discuss ideas before submitting substantial changes. A suggested workflow:

1. Fork the repo and create a branch for your feature or fix
2. Add tests and documentation for your changes
3. Open a pull request describing the motivation and impact

## License

This repository is provided for research purposes. Please add a LICENSE file (e.g., MIT, Apache-2.0) if you want to permit reuse under an explicit license. If you want me to add a LICENSE file, tell me which license to use and I will create it.


## Contact

Maintainer: Udaykirancheera15

For questions, issues, or collaboration requests, open an issue or contact the maintainer via their GitHub profile.
