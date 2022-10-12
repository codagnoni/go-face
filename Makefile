precommit: gofmt-staged

gofmt-staged:
	./gofmt-staged.sh

testdata:
	git clone https://github.com/Kagami/go-face-testdata testdata

test: testdata
	go test -v

# We don't need make's built-in rules.
MAKEFLAGS += --no-builtin-rules
# Be pedantic about undefined variables.
MAKEFLAGS += --warn-undefined-variables
.SUFFIXES:

# Used internally.  Users should pass GOOS and/or GOARCH.
OS := $(if $(GOOS),$(GOOS),$(shell go env GOOS))
ARCH := $(if $(GOARCH),$(GOARCH),$(shell go env GOARCH))
ifeq ($(OS), darwin)
  CGO_CPPFLAGS="-I/usr/local/Cellar/dlib/19.24_1/include -I/usr/local/Cellar/jpeg-turbo/2.1.4/include"
  CGO_LDFLAGS="-L/usr/local/Cellar/dlib/19.24_1/lib -L/usr/local/Cellar/jpeg-turbo/2.1.4/lib"
endif

all: build

build:
	CGO_CPPFLAGS=$(CGO_CPPFLAGS) CGO_LDFLAGS=$(CGO_LDFLAGS) go build