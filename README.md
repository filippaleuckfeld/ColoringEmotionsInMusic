# Coloring Emotions in Music: Visualizing Emotional Responses with Generative Art

![Example Visualization](images/example.png)

## Project Overview

This project aims to explore how emotions evoked by music can be translated into visual art using color. It is part of a master's thesis at KTH Royal Institute of Technology that investigates the intersection of music, emotion, and generative art.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [TouchDesigner Integration](#touchdesigner-integration)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.9.1
- Git
- Virtual environment (recommended)

### Clone the Repository

To clone the repository, run:

    git clone git@github.com:filippaleuckfeld/ColoringEmotionsInMusic.git
    cd ColoringEmotionsInMusic

### Set Up Virtual Environment

To set up the virtual environment, run:

    python -m venv venv
    source venv/bin/activate 
    
### Install Dependencies

To install the dependencies, run:

    pip install -r requirements.txt

### Data Requirements

The data required to run this project is not available in this repository due to copyright reasons. To run the project, you need a CSV file with the following data columns:

    isrc,title,spotify_uri,artist,genre_id,genre_name,category_id,category_name

## TouchDesigner Integration

The project includes a TouchDesigner `.toe` file located in the `TouchDesigner` folder. This file is used to visualize the emotional responses to music tracks through generative art.

### Running the TouchDesigner Project

1. Make sure TouchDesigner is installed on the computer you want to run it on.
2. Open the `.toe` file in TouchDesigner.
3. Ensure the audio file paths and other configurations are correctly set within the TouchDesigner project.
4. Run the project to visualize the emotional characteristics of the music tracks.

## License

This project is licensed under the [GNU General Public License v2.0 (GPL-2.0)](LICENSE).

## Contact

For any questions or comments, please contact:
Filippa Leuckfeld - [Email](mailto:efle@kth.se)
