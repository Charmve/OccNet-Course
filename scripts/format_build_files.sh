#! /bin/bash
scripts/buildifier -lint=fix $(find onboard -type f \( -iname BUILD \))
scripts/buildifier -lint=fix $(find offboard -type f \( -iname BUILD \))
