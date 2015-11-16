class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1
        if (len(conf.buffer) <= 0) or (len(conf.stack) <= 0):
            return -1
        if conf.buffer[0] == 0:
            # here is the Root element
            return -1

        idx_wi = conf.stack[len(conf.stack) - 1]

        flag = True
        for (idx_parent, r, idx_child) in conf.arcs:
            if idx_child == idx_wi:
                flag = False

        if flag:
            conf.stack.pop()
            idx_wj = conf.buffer[0]
            conf.arcs.append((idx_wj, relation, idx_wi))
        else:
            return -1

        # raise NotImplementedError('Please implement left_arc!')
        # return -1
        # idx_wb = conf.buffer.pop(0)
        # idx_ws = conf.stack[-1]

        # conf.stack.pop(-1)
        # conf.arcs.append((idx_wb, relation, idx_ws))

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        idx_wi = conf.stack[-1]
        idx_wj = conf.buffer.pop(0)

        conf.stack.append(idx_wj)
        conf.arcs.append((idx_wi, relation, idx_wj))
        return conf

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # raise NotImplementedError('Please implement reduce!')
        # return -1
        if not conf.stack or len(conf.stack) <= 0:
            return -1

        idx_wi = conf.stack[len(conf.stack) - 1]
        flag = False
        for (idx_parent, r, idx_child) in conf.arcs:
            if idx_child == idx_wi:
                flag = True
        if flag:
            conf.stack.pop()  # reduce it
        else:
            return -1
        # conf.stack.pop(-1)
        return conf

    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        # raise NotImplementedError('Please implement shift!')
        # return -1
        if not conf.buffer:
            return -1
        idx_wb = conf.buffer.pop(0)

        conf.stack.append(idx_wb)
        return conf
