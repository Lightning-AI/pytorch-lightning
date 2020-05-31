from argparse import ArgumentParser

parser1 = ArgumentParser(add_help=True)

subs = parser1.add_subparsers()
parser2 = subs.add_parser('model', parents=[parser1], add_help=False)
parser2.add_argument('--parser2', default=2)


parser3 = subs.add_parser('p3', parents=[parser1], add_help=False)
parser3.add_argument('--parser3', default=3)

args2, _ = parser2.parse_known_args()
args3, _ = parser3.parse_known_args()

print(args2)
print(args3)