# Selective Amnesia on a Conditional VAE for MNIST

This repository contains a Google Colab notebook implementing the Selective Amnesia algorithm on a CVAE for the MNIST dataset. The goal is to selectively "unlearn" or "forget" specific digit classes from a pre-trained CVAE model.

It also contains a fixed.ipynb notebook where further experiments were conducted: plotting the Latent Space and Latent Traversal Grid, and exploring the effects of different LATENT_DIM and EPOCHS values on the quality of the generated outputs.

The implementation is based on the paper:
**Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models**
(Alvin Heng, Harold Soh - arXiv:2305.10120)

---

## Tasks Completed

### Task 1: Implementing Selective Amnesia

* **Baseline CVAE Training:** A conditional VAE was trained on all 10 MNIST classes (0-9) to serve as the base model.
* **Fisher Information Calculation:** The diagonal of the Fisher Information Matrix (FIM) was calculated using only the data corresponding to the class(es) intended to be forgotten. This identifies the parameters critical for generating those specific classes.
* **Unlearning Implementation:** The core Selective Amnesia algorithm was implemented. This involves fine-tuning the baseline model using a custom loss function composed of three parts:
    1.  **$\mathcal{L}_{retain}$**: Standard VAE loss on the *retained* classes (using real data + generative replay) to prevent catastrophic forgetting.
    2.  **$\mathcal{L}_{SA}$**: The Selective Amnesia loss, which **subtracts** a penalty based on the Fisher importance, encouraging the model to move parameters away from values important for the *forget* class(es).
    3.  **$\mathcal{L}_{neg}$**: A "gray-out" loss (inspired by the paper's repository) that encourages the model to output a neutral gray image for the forgotten classes.

### Task 2: Quality Improvement & Visualization

* **Image Quality:** Attempts were made to improve the quality of the generated MNIST digits by increasing the `LATENT_DIM` of the CVAE and the number of `EPOCHS` for baseline training. While quality improved, the inherent blurriness typical of VAEs remains a challenge.
* **Visualizations Added:** To better understand the model's behavior and the effect of unlearning, several plots were implemented:
    1.  **Comparison Grid:** Shows generated samples from the original baseline model side-by-side with the unlearned model for all 10 classes. This visually confirms if the target classes have been corrupted.
    2.  **Latent Space Plot (t-SNE):** Visualizes the CVAE's latent space using t-SNE (or directly if `LATENT_DIM=2`). It shows how the model clusters different digits and how these clusters change after unlearning (forgotten classes become less distinct or "dissolved").
    3.  **Latent Traversal Grid:** Generates a grid of images by interpolating between points in the latent space and corresponding class labels. This visualizes the smoothness and organization of the learned latent manifold.

---

## How to Use the Notebook

1.  **Open in Google Colab or compatible environment.**
2.  **Run Cells Sequentially (Phase 1):** Execute the cells from "Imports" down to "visual compar func" **once**. This will train the baseline model (`vae_baseline.pt`).
3.  **Run Experiments (Phase 2):**
    * For each set of classes you want to forget:
        * Run the "Experiment" cell, setting the `FORGET_CLASSES` list.
        * Run the "FIM" cell.
        * Run the "Unlearning sess" cell. This saves the unlearned model.
        * Run the corresponding "visual output" cell to see the Comparison Grid, t-SNE plot, and Traversal plot for that experiment.
    * You can repeat this block (Config, FIM, Unlearning, Plots) for different `FORGET_CLASSES` without retraining the baseline.

---

## Observations

* The Selective Amnesia algorithm successfully corrupts the model's ability to generate the specified forgotten classes.
* The performance (effectiveness of forgetting) did not seem significantly different across various chosen forget classes.
* The quality of generated images is inherently limited by the VAE architecture, though increasing model capacity and training time helps.
* Latent space visualizations clearly show the disruption of clusters corresponding to the forgotten classes after unlearning.
