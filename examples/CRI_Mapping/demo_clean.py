#!/usr/bin/env python3

 converter = CRI_Converter(
        num_steps=10,
        input_layer=0,
        snn_layers=5,
        output_layer=14,
        v_threshold=1,
        input_shape=(2, 128, 128),
        backend="spikingjelly",
        embed_dim=0,
        dvs=True,
        converted_model_pth=converted_model_pth,
    )
    converter.layer_converter(net_quan)
    axons = dict(converter.axon_dict)
    neurons = dict(converter.neuron_dict)
    outputs = converter.output_neurons

    # breakpoint()
    hardwareNetwork = CRI_network(
        axons=axons,
        connections=neurons,
        config=config,
        target="simpleSim",
        outputs=outputs,
    )



    start_time = time.time()

    test_loss = 0
    test_acc = 0
    test_samples = 0

    writer = SummaryWriter(log_dir="./log_hardware")
    encoder = encoding.PoissonEncoder()

    loss_fun = nn.MSELoss()
    torchnet.eval()
    for img, label, x_len in tqdm(test_loader):
        # T appears to be different for different batches
        img = img.transpose(0, 1)  # [B, T, C, H, W] -> [T, B, C, H, W]
        label_onehot = F.one_hot(label, args.targets).float()
        out_fr = 0.0

        cri_input = []

        for t in img:
            encoded_img = encoder(t)
            cri_input.append(encoded_img)
            netout = torchnet(encoded_img)
            print("size" + str(len(img)))
            print(netout)

        torch_input = cri_input

        # breakpoint()
        cri_input = torch.stack(cri_input)
        # breakpoint()
        cri_input = cri_input.transpose(0, 1)
        # looks like the converter wants batch in the first dimension
        cri_input = converter.input_converter(cri_input)
        out_fr = torch.tensor(converter.run_CRI_sw(cri_input, net), dtype=float).to(
            device
        )
        # breakpoint()
        for idx, elem in enumerate(out_fr):
            row = torch.zeros_like(elem)
            hot = torch.argmax(elem)
            row[hot] = 1
            out_fr[idx] = row

        # breakpoint()

        loss = loss_fun(out_fr, label_onehot)
        test_samples += label.numel()
        test_loss += loss.item() * label.numel()

        test_acc += (out_fr.argmax(1) == label).float().sum().item()
        print("acc: " + str(test_acc / test_samples))
    # breakpoint()

    test_time = time.time()
    test_speed = test_samples / (test_time - start_time)
    test_loss /= test_samples
    test_acc /= test_samples
    breakpoint()
    writer.add_scalar("test_loss", test_loss)
    writer.add_scalar("test_acc", test_acc)

    print(f"test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}")
    print(f"test speed ={test_speed: .4f} images/s")
