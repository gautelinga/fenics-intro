import dolfin as df

class Params():
    def __init__(self, *args):
        self.dolfin_params = dict()

        if len(args) > 0:
            input_file = args[0]
            with open(input_file, "r") as infile:
                for el in infile.read().split("\n"):
                    if "=" in el:
                        key, val = el.split("=")
                        if val in ["true", "TRUE"]:
                            val = "True"
                        elif val in ["false", "FALSE"]:
                            val = "False"
                        try:
                            self.dolfin_params[key] = eval(val)
                        except:
                            self.dolfin_params[key] = val


    def __getitem__(self, key):
        if key in self.dolfin_params:
            return self.dolfin_params[key]
        else:
            exit("No such parameter: {}".format(key))
            #return None

    def __setitem__(self, key, val):
        self.dolfin_params[key] = val

    def __str__(self):
        string = ""
        for key, val in self.dolfin_params.items():
            string += "{}={}\n".format(key, val)
        return string

    def save(self, filename):
        if df.MPI.rank(df.MPI.comm_world) == 0:
            with open(filename, "w") as ofile:
                ofile.write(self.__str__())

class GenSub(df.SubDomain):
    def __init__(self, Lx, Ly):
        self.Lx = Lx
        self.Ly = Ly
        super().__init__()

class Top(GenSub):
    def inside(self, x, on_bnd):
        return on_bnd and x[1] > self.Ly - df.DOLFIN_EPS_LARGE

class Btm(GenSub):
    def inside(self, x, on_bnd):
        return on_bnd and x[1] < df.DOLFIN_EPS_LARGE

class Obst(GenSub):
    def inside(self, x, on_bnd):
        return on_bnd and x[0] > df.DOLFIN_EPS_LARGE and x[0] < self.Lx-df.DOLFIN_EPS_LARGE and \
            x[1] > df.DOLFIN_EPS_LARGE and x[1] < self.Lx - df.DOLFIN_EPS_LARGE

class Wall(GenSub):
    def inside(self, x, on_bnd):
        return on_bnd and x[0] < df.DOLFIN_EPS_LARGE or x[0] > self.Lx-df.DOLFIN_EPS_LARGE