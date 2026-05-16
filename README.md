# ML-Based Malicious HTTP Request Detection

## Project Overview
This repository contains a machine learning project developed for the "AI Methods in Threat Analysis in Computer Systems" course. The goal of this project is to build an intelligent classifier capable of analyzing raw HTTP requests and detecting malicious payloads commonly associated with web application attacks (e.g., SQL Injection, XSS, Path Traversal).

## Course Requirements & Assumptions
This project is developed under specific academic guidelines for the laboratory:
* **Team:** The project is implemented individually.
* **Core Deliverables:** The final artifacts include the source code of implemented methods/experiments and a presentation outlining the topic, assumptions, experiment plan, and results.
* **Experimentation:** The project must contain an experimental phase with at least two experiments implemented using proper practices.
* **Evaluation:** A statistical analysis of the experimental results is mandatory.
* **Originality:** All code must be fully understood and explainable by the author. Inability to explain how the code works is considered plagiarism/non-independent work and will result in a failing grade.
* **Presentation:** The final results will be presented in a brief, maximum 10-minute presentation.

## The Recognition Problem
The core of this project is framed as a **Binary Classification Problem**. 
The system is designed to evaluate incoming HTTP traffic and assign each request to one of two distinct classes:
* **Class 0 (Normal):** Benign, safe HTTP requests representing standard user behavior.
* **Class 1 (Anomalous):** Anomalous requests containing attack vectors or malicious payloads.

## Dataset
The project utilizes the **CSIC 2010 HTTP Dataset** sourced from Kaggle. It contains tens of thousands of labeled HTTP requests. This dataset is highly regarded in the cybersecurity community for evaluating Web Application Firewalls (WAF) and Intrusion Detection Systems (IDS).

## Project Phases
The implementation strictly follows the required experimental protocol defined in the course:
* **P1:** Dataset visualization and definition of the recognition problem.
* **P2 & P3:** Implementation of recognition methods.
* **P4:** Implementation of experiments.
* **P5:** Analysis and visualization of results.
* **P6:** Formulation of conclusions.
* **P7:** Presentation of work results.

## Tech Stack & Libraries
The project is built using Python and utilizes the allowed laboratory libraries:
* `numpy`
* `matplotlib`
* `scikit-learn`
* `scipy`
* `stream-learn`
* `imbalanced-learn`