# GLASSE

genetic_algorithm.py -> glasse method

graphics_runs.ipynb -> this notebook is to make some graphics for number of adversarials, number of incorrect classifications, and best fitnesses. To run this code you will need the files in the folders stat_digit_{number}.

generate_adv_images.py -> generate adversarial images - misclassification + dics loss in interval (target - 0.01, target + 0.01) - found with the genetic algorithm
(check folders -> for tsne, it should have a folde with all adversarial images from a run: adversarial_images > run_1 > mnist_format )

gen_tsne -> first generate tsne images with : python main.py -b [path to mnist dataset]/MNIST_dataset -p [path to mnist_format of that run]/mnist_format -o [path for output of tsne for that run]
         -> generate blend : python blend_generator.py 
            (check paths, it needs the outputs from main.py)

exp_original -> adversarial images generated with the original experiment


images_tsne -> blend maps and scatter plots build with gen tsne and the images in exp_original and the mnsit dataset

## Cite this project

If you use this project in your research work or publication, please cite it using the following BibTeX entry:

```bibtex
@inproceedings{10.1145/3583133.3596392,
  author = {Clare, Luana and Correia, Jo\~{a}o},
  title = {Generating Adversarial Examples through Latent Space Exploration of Generative Adversarial Networks},
  year = {2023},
  isbn = {9798400701207},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3583133.3596392},
  doi = {10.1145/3583133.3596392},
  abstract = {Artificial Neural Networks are vulnerable to adversarial examples, malicious inputs that aim to subvert neural networks' outputs. Generative Adversarial Networks (GANs) are generative models capable of generating data that follows the training data distribution. We explore the hypothesis of using the latent space of the trained GAN to find adversarial examples. We test the adversarial examples on external classifiers trained on the same training data. Thus, we propose a framework for Generating adversariaL exAmpleS through latent Space Exploration (GLASSE). A Genetic Algorithm evolves latent vectors as individuals and uses a trained GAN to generate examples to maximise a target activation value of the discriminator network. After the evolutionary process, an external classifier trained on the same dataset evaluates whether it is adversarial. The results indicate that we can optimise the objective and find adversarial examples. We tested the generated examples with models from the adversarial learning literature, showing that 82\% on average of the generated examples resulted in successful attacks. We show a t-SNE analysis of the examples, showcasing that generated adversarial examples are blended in the cluster of each belonging class and visually similar to the training dataset examples, showcasing the viability of the proposed approach.},
  booktitle = {Proceedings of the Companion Conference on Genetic and Evolutionary Computation},
  pages = {1760–1767},
  numpages = {8},
  keywords = {adversarial examples, generative adversarial networks, evolutionary computation, latent space exploration},
  location = {Lisbon, Portugal},
  series = {GECCO '23 Companion}
}
