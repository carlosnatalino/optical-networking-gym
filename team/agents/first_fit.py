from optical_networking_gym import select_first_fit_action


class FirstFitAgent:
    name = "first_fit"

    def solution(self, obs, info, env):
        mask = info.get("mask")

        if mask is None:
            mask = env.action_masks()

        if mask is None:
            raise RuntimeError("Không lấy được action mask.")

        return select_first_fit_action(mask)