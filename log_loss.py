import os


def log_loss(dict_loss, target, seq_length):
    print(f"Testing loss {dict_loss['loss']}")
    print(f"Mean RMS per frame {dict_loss['rms_']}")
    print(f"Mean RMS per frame (periodic) {dict_loss['rms_periodic']}")

    print("Writing logs")
    logs = open(os.path.join(target, "test_logs_{seq_length}.txt"), "w")
    logs.write(str(dict_loss['loss']) + " ")
    logs.write(str(dict_loss['rms_']) + " ")
    logs.write(str(dict_loss['rms_periodic']) + " ")
    logs.close()

    logs = open(os.path.join(target, f"test_logs_per_frame_{seq_length}.txt"),
                "w")
    for frame_id in range(len(dict_loss['rms_per_frame'])):
        logs.write(str(dict_loss['frame_id']) + " ")
        logs.write(str(dict_loss['rms_per_frame'][frame_id]) + " ")
        logs.write(str(dict_loss['rms_per_frame_periodic'][frame_id]) + " \n")
    logs.close()

    logs = open(
        os.path.join(target, f"test_logs_per_sequence_{seq_length}.txt"), 'w')
    for seq_id in range(len(dict_loss['rms_per_sequence'])):
        logs.write(str(seq_id) + " ")
        logs.write(str(['rms_per_sequence'][seq_id]) + " ")
        logs.write(str(dict_loss['rms_per_sequence_periodic'][seq_id]) + " \n")
    logs.close()

    logs = open(os.path.join(target, f"test_logs_lattitude_{seq_length}.txt"),
                'w')
    for l_id in range(dict_loss['rms_lattitude'].shape[0]):
        logs.write(str(dict_loss['rms_lattitude'][l_id]) + " ")
    logs.close()
