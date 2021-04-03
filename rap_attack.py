#######################################################################################################################
# RAP: Region-restricted Adversarial Perturbation, a generalized RSAEG algorithm
# Correspondinng to the following paper:
# @article{li2020black,
#   title={Black-box Attack against Handwritten Signature Verification with Region-restricted Adversarial Perturbations},
#   author={Li, Haoyang and Li, Heng and Zhang, Hansong and Yuan, Wei},
#   journal={Pattern Recognition},
#   pages={107689},
#   year={2020},
#   publisher={Elsevier}
# }
# Written by Li, Haoyang in Apr. 2021
#######################################################################################################################

import torch
import logging
import numpy as np

from target_model import MLP
from matplotlib import pyplot as plt

LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log_path = 'records/rap_attack.log'

logging.basicConfig(filename=log_path, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# The image is organized in pyTorch style, i.e. (batch, channel, height, width)

# Calculate RMSE (Root Mean Square Error)
def rmse(img1, img2):
    """
    Calculate the RMSE of image1 and image2
    :param img1: image1
    :param img2: image2
    :return: the RMSE value
    """
    res = np.sqrt(np.mean(np.power(img1-img2,2)))
    return res

# Calculate W-RMSE (Weighted Root Mean Square Error)
def wrmse(img1, img2):
    """
    Calculate the weighted-RMSE, the weight is img1 itself in this case
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    This weight matrix (i.e. img1) is not always reasonable for general images
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param img1: genuine image of signature 1
    :param img2: adversarial image of signature 1
    :return: the WRMSE value
    """
    weight = img1
    difference = (img1-img2)
    res = np.sqrt(np.mean(np.power(difference,2) * weight))
    return res

# Calculate perturbing rounds of K-Perturbations
def stepsAmplitude(prob):
    """
    Calculate the amplitude ratio for rounds of k batch perturbation to perform
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    This is a formula by experience, the rounds itself could be a parameter for this algorithm
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param prob: the initial probability
    :return: the calculated ratio
    """
    steps = np.log(1/prob)/np.log(10)
    return steps

#Binary search for a near optimal intensity of perturbations
def valueBand(v0, delta=0.05):
    """
    Get the value band of the perturbation for a pixel with intensity of v0
    :param v0: the intensity of the target pixel
    :param delta: the difference constraint
    :return: a tuple containing the low edge and high edge of the value band
    """
    vl = max(v0 - delta, 0)
    vh = min(v0 + delta, 1)
    return vl, vh

def nearOptimalIntensity(smodel, adv, maxpos, L=5, delta=0.05):
    """
    Get the near optimal intensity using a binary search
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    To make it work in RGB images, the perturbation should be a vector rather than a scalar
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param smodel: the target classifier
    :param adv: the image that turns into an adversarial example
    :param maxpos: the proposed positions to perturb
    :param L: the rounds of binary search
    :param delta: the difference constraint for each perturbation
    :return: the desired new value for pixels in maxpos
    """

    ac = np.max(adv[maxpos])
    vl, vh = valueBand(ac,delta)

    v = (vh + vl) / 2
    adv_temp = adv.copy()
    adv_temp[maxpos] = adv[maxpos]
    adv_temp[maxpos] = vl
    _,pl = smodel.predict(adv_temp)
    for div in range(L):
        adv_temp[maxpos] = adv[maxpos]
        adv_temp[maxpos] = v
        _, pv = smodel.predict(adv_temp)
        if pv > pl:
            vl = v
            v = (vl + vh) / 2
            pl = pv
        else:
            vh = v
            v = (vl + vh) / 2
    return v

#Perturb K positions chosen randomly from the stroke region
def genGuidance(img):
    """
    Generate a guide matrix to locate the positions to perturb
    In white-box scenario, this guidance could be the gradients.
    :param img: the image used as a reversed mask
    :return: the guide matrix
    """
    mask = img
    guide = np.random.randn(img.shape[2] * img.shape[3]).reshape(img.shape)
    guide = guide * mask
    return guide

def maxPosition(guide):
    """
    Get the positions to perturb according to a guide matrix
    :param guide: the guide matrix
    :return: the position to perturb
    """
    maxguide = np.max(guide)
    maxpos = np.where(guide==maxguide)
    return maxpos

def kBatchPerturbation(smodel, adv0, guidanceGenerate = genGuidance,
                       K=50, L=5, delta=0.05):
    """
    Perturb k rounds in order generate an adversarial example
    :param model: the model to attack
    :param adv0: the base adversarial example
    :param K: how many rounds to perturb ( I call it as a batch)
    :param L: how many rounds of binary search to perform in each selected pixel
    :param delta: the difference constraint for the binary search
    :return: a tuple containing
             adv: the generated adversarial example,
             prob: the probability of being genuine for this adversarial example,
             S: whether it is a success or not
    """

    adv = adv0

    guide = guidanceGenerate(adv)

    for i in range(K):
        #Acquire the next position to be perturbed
        maxpos = maxPosition(guide)
        #Perturb the pixel in deignated position
        adv[maxpos] = nearOptimalIntensity(smodel, adv, maxpos, L=L, delta=delta)
        #Mark the perturbed position as perturbed
        guide[maxpos]=0

    S, prob = smodel.predict(adv)

    return adv,prob,S

# RAP Attack
def RAPAttack(image, smodel,
              steps = 100, K = 50, L = 5, delta=0.05,
              breakprob = 0.1, startprob=0.01,
              guidanceGenerate = genGuidance):
    """
    RAPAttack algorithm, a generalized RSAEG attack
    :param image: the initial target image
    :param smodel: the target model
    :param steps: the number of attack iterations
    :param K: the batch size for each perturbation
    :param L: the rounds for binary search
    :param delta: the difference constraint for binary search
    :param breakprob: if probability is higher than breakprob
                      and the attack has gained success,
                      the attack iterations will break earlier
    :param startprob: this param is used
                      for the calculation of the amplitude ratio of steps
    :return: a tuple containing:
             adv: the generated adversarial example
             Sa: whether it is a success or not
    """

    adv0 = image.copy()
    adv = adv0

    probs = []
    rms = []
    Sa = False

    steps = int(round(steps * stepsAmplitude(startprob)))
    logging.info('Calculated steps %d' % (steps))

    for i in range(steps):
        S,prob = smodel.predict(adv)

        rm=wrmse(image,adv)

        rms.append(rm)
        probs.append(prob)

        logging.info('[Perturbation Step %d] Success: %s RMSE: %.4f Probability: %.4f'
                      % (i, str(S),rm, prob))

        # The attack is succeeded
        if S and prob > breakprob:
            break

        adv, proba, Sa = kBatchPerturbation(smodel, adv, guidanceGenerate = guidanceGenerate,
                                            K = K, L = L, delta = delta)

    plt.figure()
    plt.title('RAP Attack Process')
    plt.plot(rms, probs)
    plt.xlabel('W-RMSE')
    plt.ylabel('Probability')
    plt.savefig('records/rap_attack_process.jpg')
    plt.close()

    return adv,Sa


class SModel(object):
    """
    This is a model wrapper for easy adoption
    """
    def __init__(self, input_shape, model, target = None):
        """
        :param input_shape: the input shape of the target model, e.g. (1,1,28,28)
        :param model: the target model
        :param target: the target label, None for non-targeted attack
        """
        self.input_shape = input_shape
        self.model = model
        self.target = target

    def predict(self, x):
        x = x.reshape(self.input_shape)
        pred = self.model.predict(x)

        if self.target:
            # For targetted attack, the prob is the probability given by the model
            prob = pred[0,self.target].item()
        else:
            # For non-targeted attack, the prob is reversed probability
            prob = 1 - pred.max()

        S = bool(prob > 0.8) # Define Success
        return S, prob

if __name__ == '__main__':
    # Load the model
    model = MLP((1, 28, 28), 10)
    with open("records/mnist_model.pth", "rb") as f:
        model.load_state_dict(torch.load(f))
    model.eval()

    # Load the original example
    image = plt.imread("records/origin_example.jpg")[:, :, 0]
    image = image.reshape((1, 1, 28, 28))
    image = image / 255

    # Set up the target and model wrapper
    target_label = 5
    smodel = SModel((1, 1, 28, 28), model, target = target_label)

    # Launch RAP attack
    adv, Sa = RAPAttack(image, smodel)

    # Display generated adversarial example and corresponding adversarial perturbations
    origin_prediction = model.predict(image).argmax(-1).item()
    adversarial_prediction = model.predict(adv).argmax(-1).item()

    perturbation = (adv - image)[0, 0]
    adv = adv[0,0]
    image = image[0, 0]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Prediction ' + str(origin_prediction))
    plt.subplot(1, 3, 2)
    plt.title('Perturbations')
    plt.imshow(perturbation, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.title('Adversarial Example')
    plt.imshow(adv, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Prediction ' + str(adversarial_prediction))
    plt.savefig("records/rap_attack_result.jpg")
    plt.show()
