import torch
import numpy as np

def gaussian_mle(gt, pred):
    '''
    Define Gaussian MLE
    
    gt   : (n_samples, out_dim)
    pred : (n_samples, out_dim*2)
    
    each feature follow a different gaussian distribution,
    so the dimension for pred is 2*out_dim

    min -log(p(y|x)) where p(y)~N(miu, sigma)
    reference: http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    '''

    assert gt.size(1)*2 == pred.size(1), "size gt {} and pred {} must match".format(gt.size(), pred.size())

    n_dims = gt.size(1)

    mu = pred[:, :n_dims]
    logsigma = pred[:, n_dims:]

    '''
    after logorithmic:
    $-\frac{N}{2} \log(2\pi\sigma^2) -\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n - \mu)^2 $
    
    '''

    log2pi = -0.5*n_dims*np.log(2*np.pi)
    sigma_trace = -torch.sum(logsigma, axis=1)
    mse = -0.5*torch.sum( torch.square((gt - mu) / torch.exp(logsigma)) ,axis=1)

    nll = - ( log2pi + sigma_trace + mse )

    return torch.mean(nll), mu, logsigma


def sampler(mu, logsigma, size=10):
    '''
    Given mu and sigma, sampling from the distributions
    '''

    mu = mu.numpy()
    logsigma = logsigma.numpy()

    sigma = np.exp(logsigma)

    result = []
    for i in range(mu.shape[0]):
        mu_ = mu[i]
        sigma_ = sigma[i]
        cov = np.diag(sigma_)
        out = np.random.multivariate_normal(mu_, cov, 1)
        result.append(out)

    return np.concatenate(result, axis=0)




if __name__ == '__main__':
    
    n_samples = 4
    out_dim = 2


    # gt is target
    # pred is the results of a model. e.g., pred = model(x) where x is the input.
    gt = torch.randn(n_samples, out_dim)
    pred = torch.randn(n_samples, 2*out_dim)

    loss, mu, logsigma = gaussian_mle(gt, pred)
    print(loss)

    sample = sampler(mu, logsigma)
    print(sample)





