from ogutils.infer import InferSession

if __name__ == '__main__':

    infer_session = InferSession(
        device='cuda:2',
        only_key_frames=True
    )
    infer_session.run()