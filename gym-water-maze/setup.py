from setuptools import find_packages, setup

setup(
    name="gym_water_maze",
    description="Continuous water maze environment integrated with OpenAI/Gym",
    version="0.0.0",
    packages=["gym_water_maze"],
    install_requires = ["gymnasium","numpy", "matplotlib"],
    extras_require={
        'test': [
            "stable-baselines3"
        ]}
)
