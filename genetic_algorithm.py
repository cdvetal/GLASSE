from deap import creator, base, tools, algorithms
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as k
import statistics
import csv
import os
import time

seed = 0

"""Functions"""

def fitness(individual):
    return individual.fitness.value,

def discriminator_loss(fake_output):
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
  return fake_loss

def evaluateInd(individual):
  vector = np.array([individual])
  noise = k.constant(vector)
  label = tf.keras.utils.to_categorical([number], num_classes)
  label = tf.cast(label, tf.float32)
  noise_and_labels = tf.concat([noise, label], 1)
  fake_image = cond_gan.generator.predict(noise_and_labels,  verbose=False)
  fake_image_and_labels = tf.concat([fake_image, image_one_hot_labels], -1)
  fake_output = cond_gan.discriminator.predict(fake_image_and_labels,  verbose=False)
  total_loss = discriminator_loss(fake_output)
  value = 1 - abs(total_loss.numpy() - TARGET),
  losses.append((value, total_loss.numpy()))
  return value

def make_images_from_generation(current_gen, pop, pop_numbers):
  # makes images (individuals and activations) for entire generation
  
  predicted_number = -1
  activation = -1

  for ind in pop_numbers:
    vector = np.array([pop[ind]])
    noise = k.constant(vector)
    noise_and_labels = tf.concat([noise, label], 1)
    
    # ind image
    test_image = cond_gan.generator.predict(noise_and_labels,  verbose=False)
    plt.figure()
    plt.imshow(test_image[0, :, :, 0], cmap='gray')
    # uncomment to save individual image
    # plt.savefig(f"{path_images_generation}/gen_{current_gen}_ind_{ind}")
    plt.close()

    prediction = classif.predict(test_image,  verbose=False)
    
    # uncomment to save classifier activations image
    # plt.figure()
    # plt.bar(v, prediction[0], color ='maroon',width = 0.4)
    # plt.ylim(0, 1)
    # plt.plot(v, [0.5]*10, '--k')
    # plt.savefig(f"{path_images_classifier}/gen_{current_gen}_ind_{ind}")
    # plt.close()

    if ind == 0:
      predicted_number = np.argmax(prediction)
      activation = np.max(prediction)

    list_predictions[ind] = np.argmax(prediction)
    list_activations[ind] = np.max(prediction)

    if list_predictions[ind] != number and list_activations[ind] >= 0.5:
      adv_example = [number, list_predictions[ind], list_activations[ind], current_gen, ind, losses[ind][1], pop[ind]]
      writer_adv.writerow(adv_example)
  
  return (predicted_number, activation)

def gen_graphic(current_gen, smallest, biggest, mean):

  list_gen = list(range(0, current_gen+1))
  sm = np.reshape(smallest, (current_gen+1, ))
  bg = np.reshape(biggest, (current_gen+1, ))
  me = np.reshape(mean, (current_gen+1, ))

  plt.figure()
  if current_gen == 0:
    plt.plot(list_gen, sm, '-x', label="Smallests", color="red")
    plt.plot(list_gen, bg, '-x', label="Biggests", color="green")
    plt.plot(list_gen, me,  '-x', label="Average", color="yellow")
  else:
    plt.plot(list_gen, sm, label="Smallests", color="red")
    plt.plot(list_gen, bg, label="Biggests", color="green")
    plt.plot(list_gen, me, label="Average", color="yellow")
      

  plt.xlabel("Generation")
  plt.ylabel("Fitness")
  plt.xlim(0, generations)
  plt.legend(["Smallest fitness value in generation", "Biggest fitness value in generation", "Mean fitness value in generation"])
  plt.savefig(f'{path_graphic}/grafico_{current_gen}.png')
  plt.close()

"""Classifier"""

path_weights = './weights'
num_classes = 10
# importar classificador 
classif = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28, 28,1)),
    tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

classif.load_weights(f'{path_weights}/classifier_weights.h5')

"""Conditional GAN"""

# Conditional GAN
batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes

# Create the discriminator.
discriminator = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

class ConditionalGAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)

cond_gan.generator.load_weights(f'{path_weights}/cond_gan_gen_weights.h5')
cond_gan.discriminator.load_weights(f'{path_weights}/cond_gan_dis_weights.h5')

"""Parametros do GA"""

# Genetic algorithm
npop = 100
n = latent_dim
TARGET = 0.5
CXPB = 1
MUTPB = 1
generations = 40
mu = 0
sigma = 3
indpb = 0.1
tournsize = 3
ELITE_SIZE = 1

"""Ciclo for para runs e digitos"""

# Parent Directory path
new_folder = "./output/"
mode = 0o777

# Path
for number in range(0,10):
  stat_folder_digit = os.path.join(new_folder, f"stat_digit_{number}")
  os.mkdir(stat_folder_digit, mode)
  for run in range(1,16):
    start = time.time()

    print(run)

    time_file = f"{new_folder}/time.csv"
    f_time = open(time_file, 'a')
    writer_time = csv.writer(f_time)

    results_folder = os.path.join(new_folder, f"results_digit_{number}_run_{run}")
    os.mkdir(results_folder, mode)
    images_folder = os.path.join(results_folder, "images")
    os.mkdir(images_folder, mode)
    path_images_classifier = os.path.join(images_folder, "classifier")
    os.mkdir(path_images_classifier, mode)
    path_images_generation = os.path.join(images_folder, "generation")
    os.mkdir(path_images_generation, mode)
    path_vectors = os.path.join(results_folder, "vectors")
    os.mkdir(path_vectors, mode)
    path_graphic = os.path.join(results_folder, "graphics")
    os.mkdir(path_graphic, mode)
    path_summary = os.path.join(results_folder, "summary")
    os.mkdir(path_summary, mode)

    path_statinfo = os.path.join(stat_folder_digit, f"Arquivo_digit_{number}_run_{run}.csv")
    path_fit = os.path.join(stat_folder_digit, f"Fit_digit_{number}_run_{run}.csv")

    # save parameters
    path_param = f"{path_summary}/param.csv"
    f_param = open(path_param, 'w')
    writer_param = csv.writer(f_param)
    header_param = ['number', 'latent_dim', 'npop', 'target', 'elite size', 'ngen','cxpb', 'mutpb','mu','sigma', 'indpb','tournsize']
    writer_param.writerow(header_param)
    param_list = [number, latent_dim, npop, TARGET, ELITE_SIZE, generations, CXPB, MUTPB, mu, sigma, indpb, tournsize]
    writer_param.writerow(param_list)
    f_param.close()

    # Add dummy dimensions to the labels so that they can be concatenated with
    # the images. This is for the discriminator.
    label = tf.keras.utils.to_categorical([number], num_classes)
    image_one_hot_labels = label[:, :, None, None]
    image_one_hot_labels = tf.repeat(image_one_hot_labels, repeats=[image_size * image_size])
    image_one_hot_labels = tf.reshape(image_one_hot_labels, (-1, image_size, image_size, num_classes))

    losses = []
    count_inc_pred = [0]*generations
    count_adv = [0]*generations
    count_in_interval = [0]*generations

    # Initialization:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_flt", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_flt, n)
    toolbox.register("Population", tools.initRepeat, list, toolbox.individual)
    population = toolbox.Population(n=npop)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize, fit_attr='fitness') 
    toolbox.register("evaluate", evaluateInd)

    smallest = []
    biggest = []
    mean = []
    v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    header = ['ind_number', 'fitness value', 'abs(fake_loss-target)', 'fake_loss','number', 'predicted number', 'activation',  'vector']

    path_all = f"{path_summary}/summary.csv"
    f_all = open(path_all, 'w')
    writer_all = csv.writer(f_all)

    header_all = ['generation', 'fitness smallest', 'fitness biggest', 'fitness mean', 'fitness std', 'fake_loss worst', 'fake_loss best', 'fake_loss mean', 'fake_loss std', 'right number', 'choosen number', 'activation', 'best vec', 'centroid vec']
    writer_all.writerow(header_all)

    path_adv = f"{path_summary}/adv.csv"
    f_adv = open(path_adv, 'w')
    writer_adv = csv.writer(f_adv)

    header_adv = ["number", "predicted number", "activation", "generation","ind", "fake_loss", "vector"]
    writer_adv.writerow(header_adv)

    f_stat = open(path_statinfo, 'w')
    writer_stat = csv.writer(f_stat)
    header_stat = ['number', 'run'	,'geração',	'numero de classificações erradas',	'numero de disc loss no intervalo',	'numero de adversariais']
    writer_stat.writerow(header_stat)

    f_fit = open(path_fit, 'w')
    writer_fit = csv.writer(f_fit)
    writer_fit.writerow(['fitness smallest', 'fitness biggest', 'fitness mean', 'fitness std'])

    for g in range(0, generations):

      list_predictions = ['']*100
      list_activations = ['']*100
      path = f"{path_vectors}/gen{g}.csv"
      f = open(path, 'w')
      writer = csv.writer(f)
      writer.writerow(header)

      if g > 0:
        losses = [losses[0]]
        # Select and clone the next generation individuals
        offspring = toolbox.select(population, len(population) - ELITE_SIZE)
        # offspring = map(toolbox.clone, offspring)

        # Aplly mutation and crossover on the offspring
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
          ind.fitness.values = fit
        
        # Select elite from population, rest from offspring
        population.sort(key=lambda x: x.fitness, reverse=True)
        population = population[:ELITE_SIZE] + offspring
        population.sort(key=lambda x: x.fitness, reverse=True)

      else:
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
          ind.fitness.values = fit
        population.sort(key=lambda x: x.fitness, reverse=True)

      losses.sort(key=lambda x: x[0], reverse=True)
      pop_numbers = list(range(npop))  
      (predicted_number, activation) = make_images_from_generation(g, population, pop_numbers)
      
      for ind in range(npop):
        info = [ind, population[ind].fitness.values[0], 1 - population[ind].fitness.values[0], losses[ind][1], number, list_predictions[ind], list_activations[ind], population[ind]]
        writer.writerow(info)

      bigInd = population[0].fitness.values
      biggest.append(bigInd)
      smallInd = population[npop-1].fitness.values
      smallest.append(smallInd)
      print(f"\nGen {g}")
      print(f"Fitness value smallest: {smallInd} \tFitness value biggest {bigInd}")
      m = sum(ind.fitness.values[0] for ind in population)/npop
      mean.append(m)

      writer.writerow([""])
      writer.writerow(['fitness smallest', 'fitness biggest', 'fitness mean', 'fitness std'])
      row_fitness = [smallInd[0], bigInd[0], m, np.std(list(ind.fitness.values[0] for ind in population))]
      writer.writerow(row_fitness)

      writer.writerow(['fake_loss worst', 'fake_loss best', 'fake_loss mean', 'fake_loss std'])
      row_fake_loss = [losses[npop-1][1], losses[0][1], sum(x[1] for x in losses)/npop, np.std(list(x[1] for x in losses))]
      writer.writerow(row_fake_loss)

      row_all = [g]
      row_all.extend(row_fitness)
      row_all.extend(row_fake_loss)
      row_all.extend([number, predicted_number, activation])
      row_all.append(population[0])
      centroid = [sum(sub_list) / len(sub_list) for sub_list in zip(*population)]
      row_all.append(centroid)
      writer_all.writerow(row_all)

      f.close()

      gen_graphic(g, smallest, biggest, mean)

      for i in range(npop):
        if list_predictions[i] != number and list_activations[i] >= 0.5:
          count_inc_pred[g] = count_inc_pred[g] + 1
          if (1 - population[i].fitness.values[0]) < 0.01:
            count_adv[g] = count_adv[g] + 1
        if (1 - population[i].fitness.values[0]) < 0.01:
            count_in_interval[g] = count_in_interval[g] + 1

      info = [number, run, g, count_inc_pred[g], count_in_interval[g], count_adv[g]]
      writer_stat.writerow(info)

      writer_fit.writerow(row_fitness)

    f_all.close()
    f_adv.close()
    f_stat.close()
    f_fit.close()

    end = time.time()
    elapsed_time = end - start
    row_time = [elapsed_time]
    writer_time.writerow(row_time)
    f_time.close()

    seed = seed + 1
