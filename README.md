<div align="center">

#

<img src="assets/senoee.png" width="200">

# Senoee - Technical test

</div>

## Description

This project is a technical test for a internship position. The goal is to create a small project using NLP.

## Context

The goal of this project is to create a template that can standardize the name of any equipment in a plant. When a plant uses equipment from different manufacturers, there may be variations in the way the names are written or labeled. This can lead to communication errors or maintenance issues. With an equipment name standardization model, plant workers can be sure they are all talking about the same thing, regardless of the manufacturer or the variation in how the names are written.

## Dataset

The dataset used is in the `data` folder.

It contains 1 file: `extract_normalised_name_fr_training_data.xlsx` which contains 2 columns: `designation_fr` and `normalised_name`.

The `designation_fr` column contains the name of any equipment in a factory.

The `normalised_name` column contains the name of the equipment normalized.

## Deliverables

The project must be delivered in this git repository.

You need use `python` to create your project.

You need to provide a `predict.py` file which contains a function `predict` that takes a string as parameter (the name of the equipment) and returns a list of tuple (predicted name, confidence) or None if the name is not recognized.

## Documentation

You need to provide a `INSTRUCTIONS.md` file which contains the following information:

- How to install the project
  - List of dependencies with their version
  
- How to use the project

#

<p align='center'>
    <a href="https://www.linkedin.com/company/senoee/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
    </a>
    <a href="https://www.senoee.com/">
        <img src="https://img.shields.io/badge/WebSite-1a2b6d?style=for-the-badge&logo=GitHub Sponsors&logoColor=white">
    </a>
</p>

> :rocket: Don't hesitate to contact us if you have any questions or suggestions.