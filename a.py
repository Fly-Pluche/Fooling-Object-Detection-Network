 a.py
def forward(self, adv_patch, boxes_batch, lab_batch, people_id=0):
    # make people id be 1 other is 0
    for i in range(lab_batch.size()[0]):
        for j in range(lab_batch.size()[1]):
            if lab_batch[i, j] == people_id:
                lab_batch[i, j] = 1
            else:
                lab_batch[i, j] = 0
    # make a batch of adversarial patch
    adv_patch = self.median_pooler(adv_patch.unsqueeze(0))
    # determine size of padding
    img_size = np.array(self.configs.img_size)
    # adv_patch is a square
    pad = list((img_size - adv_patch.size(-1)) / 2)
    # a image needs boxes number patch
    adv_batch = adv_patch.unsqueeze(0)
    # a batch adv_batch: (batch size, boxes number, patch.size(0),patch.size(1),patch.size(2))
    adv_batch = adv_batch.expand(boxes_batch.size(0), boxes_batch.size(1), -1, -1, -1)
    batch_size = torch.Size((boxes_batch.size(0), boxes_batch.size(1)))

    # Create random contrast tensor
    contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
    contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
    contrast = contrast.cuda()

    # Create random brightness tensor
    brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
    brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
    brightness = brightness.cuda()

    # Create random noise tensor
    noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

    # apply contrast, brightness and clamp
    adv_batch = adv_batch * contrast + brightness + noise
    adv_batch = torch.clamp(adv_batch, 0.000001, 0.999999)

    # pad patch and mask to image dimensions
    my_pad = nn.ConstantPad2d((int(pad[0] + 0.5), int(pad[0]), int(pad[1] + 0.5), int(pad[1])), 0)
    adv_batch = my_pad(adv_batch)

    # rotation and rescaling transforms
    angle_size = (lab_batch.size(0) * lab_batch.size(1))
    angle = torch.cuda.FloatTensor(angle_size).uniform_(self.min_angle, self.max_angle)

    # Resizes and rotates the patch
    current_patch_size = adv_patch.size(-1)
    boxes_batch_scaled = torch.cuda.FloatTensor(boxes_batch.size()).fill_(0)
    img_size = list(img_size)
    # box [x,y,w,h]
    boxes_batch_scaled[:, :, 0] = boxes_batch[:, :, 0] * img_size[0]
    boxes_batch_scaled[:, :, 1] = boxes_batch[:, :, 1] * img_size[1]
    boxes_batch_scaled[:, :, 2] = boxes_batch[:, :, 2] * img_size[0]
    boxes_batch_scaled[:, :, 3] = boxes_batch[:, :, 3] * img_size[1]
    target_size = torch.sqrt_(
        ((boxes_batch_scaled[:, :, 2].mul(0.3)) ** 2) + ((boxes_batch_scaled[:, :, 3].mul(0.3)) ** 2)
    )
    target_x = boxes_batch[:, :, 0].view(np.prod(batch_size))
    target_y = boxes_batch[:, :, 1].view(np.prod(batch_size))
    target_off_x = boxes_batch[:, :, 2].view(np.prod(batch_size))
    target_off_y = boxes_batch[:, :, 3].view(np.prod(batch_size))

    # random change the patches' position
    off_x = target_off_x * (torch.cuda.FloatTensor(target_off_x.size()).uniform_(-0.3, 0.3))
    target_x = target_x + off_x
    off_y = target_off_y * (torch.cuda.FloatTensor(target_off_y.size()).uniform_(-0.1, 0.1))
    target_y = target_y + off_y
    target_y = target_y - 0.01
    scale = target_size / current_patch_size
    scale = scale.view(angle_size)

    s = adv_batch.size()
    adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])

    tx = (-target_x + 0.5) * 2
    ty = (-target_y + 0.5) * 2
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    # theta = roataion, rescale matrix
    theta = torch.cuda.FloatTensor(angle_size, 2, 3).fill_(0)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = sin / scale
    theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
    theta[:, 1, 0] = -sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

    grid = F.affine_grid(theta, adv_batch.shape)
    adv_batch_t = F.grid_sample(adv_batch, grid, align_corners=True)
    adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])

    adv_batch_t = torch.clamp(adv_batch_t, 0, 0.999999)
    return adv_batch_t

