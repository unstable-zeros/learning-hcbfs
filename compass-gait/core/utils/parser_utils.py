from munch import Munch


def add_cg_arguments(parser):
    parser.add_argument('--hip-mass', type=float, default=10.0,
                            help='Hip mass of compass gait walker.')
    parser.add_argument('--leg-mass', type=float, default=5.0,
                            help='Leg mass of compass gait walker.')
    parser.add_argument('--leg-length', type=float, default=1.0,
                            help='Leg length of compass gait walker.')
    parser.add_argument('--leg-cent-of-mass', type=float, default=0.5,
                            help='Center of mass of compass gait walker.')
    parser.add_argument('--gravity', type=float, default=9.81,
                            help='Gravitional constant.')
    parser.add_argument('--slope', type=float, default=0.0525,
                            help='Slope of ramp for compass gait walker.')

def munch_cg_params(args):
    args.cg_params = Munch(
        mass_hip=args.hip_mass,
        mass_leg=args.leg_mass,
        length_leg=args.leg_length,
        center_of_mass_leg=args.leg_cent_of_mass,
        gravity=args.gravity,
        slope=args.slope
    )