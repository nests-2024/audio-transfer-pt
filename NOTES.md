mcnn = RandomCNN(out_channels=16, kernel=(11, 3))
result = run_transfer(mcnn, content_s, style_s, num_steps=2000, content_weight=1e-1, style_weight=1e2)

- styleLoss:
    - G / W: content_weight=1e-1, style_weight=1e2: 48s
    - G / (C * H * W): num_steps=1000, content_weight=1e-1, style_weight=1e10: 24s
    - MSE / W²: content_weight=1e-1, style_weight=1e2: 45s
    - MSE / (C*H*W): content_weight=1e-1, style_weight=1e4: 45s
    + MSE / (C*H*W)²: num_steps=1000 content_weight=1e-1, style_weight=1e10: 21s

+ ReLU

:= ReLU is good, or no difference
:= MSE / CHW2 converges faster

------------------------------------------------------------------
- H as C, or....
+ kernel of size (H x [1,3]) and 8192 Conv2D filters and NFFT = 4096


mcnn = RandomCNN(out_channels=9000, kernel=(content_s.shape[0], 1))
result = run_transfer(mcnn, content_s, style_s, num_steps=1000, content_weight=5e-2, style_weight=1e8)

mcnn = RandomCNN(out_channels=9000, kernel=(content_s.shape[0], 11))
result = run_transfer(mcnn, content_s, style_s, num_steps=1000, content_weight=5e-2, style_weight=1e8)

:= no big difference

------------------------------------------------------------
+ Conv2D with more inputs, and stride > 1

kx = [11,17]
ky = 5
mcnn = RandomCNN(out_channels=384, kernel=(kx, ky), stride=(kx-2, ky-2))
result = run_transfer(mcnn, content_s, style_s, num_steps=1500, content_weight=1e-1, style_weight=1e10)

:= better

---------------------------------------------------------------------
- Conv2D with different sized kernels

--------------------------------------------------------------------
--------------------------------------------------------------------
TEXTURES

- with and without avg spectrogram for content

kx = 17
ky = [1,3,5]

mcnn = RandomCNN(out_channels=384, kernel=(kx, ky), stride=(kx - 2, ky - 2))
result = run_transfer(mcnn, content_s, style_s, num_steps=1500, content_weight=1e-1, style_weight=1e10)

- without content
style_weight=[1e8 - 1e12]
result = run_transfer(mcnn, content_s, style_s, num_steps=1000, content_weight=0, style_weight=1e8)


- seed
    - grab random sound and make avg spectrogram
    - repeat, reflect to 15s
    - combine multiple input files for style

    - synthesize random phrase using human voice based on seed
    - modulate bird sound with voice


+ https://huggingface.co/facebook/mms-tts-por
+ https://huggingface.co/facebook/mms-tts-spa
