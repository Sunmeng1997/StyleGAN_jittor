import jittor as jt
from Model import StyledGenerator, Discriminator
import jittor.transform as transform
from dataset import SymbolDataset
from tqdm import tqdm
import math
import random
jt.flags.use_cuda = True
jt.flags.log_silent = True

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    init_size = 8
    max_size  = 64
    dataset_path = '../color_symbol_7k'
    init_step = int(math.log2(init_size) - 2)
    max_step  = int(math.log2(max_size) - 2)
    nsteps = max_step - init_step + 1
    lr = 1e-3
    mixing = True
    code_size = 512
    batch_size = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32 }
    batch_default = 32
    phase = 150000
    max_iter = 10000

    transform = transform.Compose([
        transform.ToPILImage(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    netG = StyledGenerator(code_dim=code_size)
    netD = Discriminator(from_rgb_activate=True)

    d_optimizer = jt.optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer = jt.optim.Adam(netG.generator.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({
        'params': netG.style.parameters(),
        'lr': lr * 0.01,
        'mult': 0.01,})

    step = init_step
    resolution = int(4 * 2 ** step)
    image_loader = SymbolDataset(dataset_path, transform, resolution).set_attrs(batch_size=batch_size.get(resolution, batch_default),shuffle=True)
    train_loader = iter(image_loader)

    requires_grad(netG, False)
    requires_grad(netD, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0
    final_progress = False
    pbar = tqdm(range(max_iter))

    for i in range(max_iter):
        alpha = min(1, 1 / phase * (used_sample + 1))
        if (resolution == init_size) or final_progress:
            alpha = 1

        if used_sample > phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1
            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            image_loader = SymbolDataset(dataset_path, transform, resolution).set_attrs(
                batch_size=batch_size.get(resolution, batch_default),
                shuffle=True
            )
            train_loader = iter(image_loader)

        try:
            real_image = next(train_loader)
        except (OSError, StopIteration):
            train_loader = iter(image_loader)
            real_image = next(train_loader)

        real_image.requires_grad = True
        b_size = real_image.size(0)

        real_scores = netD(real_image, step=step, alpha=alpha)
        real_predict = jt.nn.softplus(-real_scores).mean()

        grad_real = jt.grad(real_scores.sum(), real_image)
        grad_penalty = ( grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2 ).mean()
        grad_penalty = 10 / 2 * grad_penalty

        if i % 10 == 0:
            grad_loss_val = grad_penalty.item()

        if mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(4, b_size, code_size).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        else:
            gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = netG(gen_in1, step=step, alpha=alpha)
        fake_predict = netD(fake_image, step=step, alpha=alpha)
        fake_predict = jt.nn.softplus(fake_predict).mean()

        if i % 10 == 0:
            disc_loss_val = (real_predict + fake_predict).item()

        loss_D = real_predict + grad_penalty + fake_predict
        d_optimizer.step(loss_D)

        requires_grad(netG, True)
        requires_grad(netD, False)

        fake_image = netG(gen_in2, step=step, alpha=alpha)
        predict = netD(fake_image, step=step, alpha=alpha)
        loss_G = jt.nn.softplus(-predict).mean()

        if i % 10 == 0:
            gen_loss_val = loss_G.item()

        g_optimizer.step(loss_G)

        requires_grad(netG, False)
        requires_grad(netD, True)

        used_sample += real_image.shape[0]

        if i % 10000 == 0:
            jt.save(netG.state_dict(), f'./checkpoint/{str(i + 1).zfill(6)}.pkl')

        if i%100 ==0:
            print('Resolution: ',resolution, 'G_loss: ',gen_loss_val,' D_loss: ' , disc_loss_val)



