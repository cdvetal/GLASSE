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
