# libCEED: How to Contribute

Contributions to libCEED are encouraged.
<!---
Please use a pull request to the appropriate branch ('maint' for
backward-compatible bug fixes for the last stable release, 'master' for
new features and everything else).
-->
Please make your commits well-organized and
[atomic](https://en.wikipedia.org/wiki/Atomic_commit#Atomic_commit_convention),
using `git rebase --interactive` as needed.  Check that tests
(including "examples") pass using `make prove-all`.  If adding a new
feature, please add or extend a test so that your new feature is
tested.

In typical development, every commit should compile, be covered by the
test suite, and pass all tests.  This improves the efficiency of
reviewing and facilitates use of
[`git bisect`](https://git-scm.com/docs/git-bisect).

Open an issue or RFC (request for comments) pull request to discuss
any significant changes before investing time.  It is useful to create
a WIP (work in progress) pull request for any long-running development
so that others can be aware of your work and help to avoid creating
merge conflicts.

Write commit messages for a reviewer of your pull request and for a
future developer (maybe you) that bisects and finds that a bug was
introduced in your commit.  The assumptions that are clear in your
mind while committing are likely not in the mind of whomever (possibly
you) needs to understand it in the future.

Give credit where credit is due using tags such as `Reported-by:
Helpful User <helpful@example.com>`.  Please use a real name and email
for your author information (`git config user.name` and `user.email`).

Please avoid "merging from upstream" (like merging 'master' into your
feature branch) unless there is a specific reason to do so, in which
case you should explain why in the merge commit.
[Rationale](https://lwn.net/Articles/328436/) from
[Junio](https://gitster.livejournal.com/42247.html) and
[Linus](http://yarchive.net/comp/linux/git_merges_from_upstream.html).

You can use `make style` to help conform to coding conventions of the
project, but try to avoid mixing whitespace or formatting changes with
content changes (see atomicity above).

By submitting a pull request, you are affirming the following.

## [Developer's Certificate of Origin 1.1](https://developercertificate.org/)

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.


