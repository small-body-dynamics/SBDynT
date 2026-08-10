import argparse


def main():
    description = "The Small Body Dynamics Tool"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--version",
        help="Print version information",
        dest="version",
        action="store_true",
    )

    args = parser.parse_args()

    if args.version:
        import sbdynt

        print(sbdynt.__version__)
        return


if __name__ == "__main__":
    main()
