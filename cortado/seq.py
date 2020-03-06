
class Seq:
    def __init__(self, gen_fun):
        self.gen_fun = gen_fun

    @classmethod
    def from_next(cls, state, next):
        return cls(lambda : (state, next))

    @classmethod
    def from_gen(cls, gen):
        def f(gen):
            try:
                v = next(gen)
                return v, gen
            except StopIteration:
                return None, None
        return cls(lambda : (gen, f))

    @staticmethod
    def try_read(xs):
        state, next = xs.gen_fun()
        v, new_state = next(state)
        if v is None:
            return None, Seq.from_next(None, (lambda s: None, None))
        else:
            return v, Seq.from_next(new_state, next)

    @staticmethod
    def map(f, xs):
        def new_gen_fun():
            state, next = xs.gen_fun()
            def new_next(s):
                v, new_state = next(s)
                if v is None:
                    return None, s
                else:
                    return f(v), new_state
            return state, new_next
        return Seq(new_gen_fun)

    @staticmethod
    def reduce(f, acc0, xs):
        eof = False
        acc = acc0
        tail = xs
        while not eof:
            v, tail = Seq.try_read(tail)
            if v is None:
                eof = True
            else:
                acc = f(acc, v)
        return acc

    @staticmethod
    def zip(*xss):
        def new_gen_fun():
            y = tuple(map((lambda x: x.gen_fun()), xss))
            state = tuple(map((lambda x: x[0]), y))
            next = tuple(map((lambda x: x[1]), y))
            def new_next(s):
                res = tuple(map((lambda x: x[1](x[0])), zip(s, next))) 
                newstate = tuple(map((lambda x: x[1]), res))
                if all([a is not None for (a, b) in res]):
                    v = tuple(map((lambda x: x[0]), res))
                    return v, newstate
                else:
                    return None, newstate
            return state, new_next
        return Seq(new_gen_fun)

    @staticmethod
    def foreach(f, xs):
        eof = False
        tail = xs
        while not eof:
            v, tail = Seq.try_read(tail)
            if v is None:
                eof = True
            else:
                f(v)

    @staticmethod
    def take(xs, n):
        def new_gen_fun():
            state, next = xs.gen_fun()
            def f(s):
                state = s[0]
                n = s[1]
                if n >= 1:
                    v, newstate = next(state)
                    return v, (newstate, n - 1)
                else:
                    return None, state
            return (state, n), f
        return Seq(new_gen_fun)

    @staticmethod
    def tolist(xs):
        res = []
        eof = False
        tail = xs
        while not eof:
            v, tail = Seq.try_read(tail)
            if v is None:
                eof = True
            else:
                res.append(v)
        return res












