z = torch.randn([1,latent_size]).repeat(1,1).to(device)
z_noise = noise_scale * torch.randn([1,latent_size]).to(device)
z_perturb = z + z_noise
if in_W:
    w = G.style_gan2.get_latent(z_perturb)
    img_orig = G.style_gan2([w] , input_is_latent=True)[0]
else:
    w = z_perturb
    img_orig = G(w)

if gan_resolution != classifier_input_size:
    img_orig = resize_transform(img_orig)

y_orig = classifier(img_orig)
y_orig = torch.sigmoid(y_orig[0])  
y_target = ~(y_orig[:,subset_labels]>0.5)#torch.randint(-1, 2, y_orig[:,subset_labels].shape).to(device)
y_target = y_target.int()
loss_filter = abs(y_target) > 0


ig = img_orig[0].permute(1, 2, 0).cpu().detach().numpy()
plt.imshow(ig )
plt.axis('off')

dir_pred = shift_model(w,y_target)
if in_W:
    img_shift = G.style_gan2([w + dir_pred] , input_is_latent=True)[0]
else:
    img_shift = G(w + dir_pred)


ig = img_shift[0].permute(1, 2, 0).cpu().detach().numpy()
plt.imshow(ig )
plt.axis('off')

y_shift = classifier(img_shift)
print(y_shift)
if isinstance(y_shift,tuple):
    y_shift = y_shift[0]
print(y_shift)
y_out = torch.sigmoid(y_shift)
print(y_out)
y_target = (y_target + 1.0) / 2.0 # map from [-1 0 1] to [0.0 0.5 1.0] for loss computation purposes
print(y_target)
if len(subset_labels) > 0:
    y_out = y_out[:,subset_labels]
print(y_out)
dir_loss = criterion(y_out[loss_filter],y_target[loss_filter].float())