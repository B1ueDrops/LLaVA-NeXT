from ogutils.infer import InferSession

if __name__ == '__main__':

    infer_session = InferSession(
        device='cuda:1',
        only_key_frames=False
    )
    infer_session.run()