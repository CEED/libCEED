# How to Contribute

Contributions to libCEED are welcome.

Please use a pull request to the appropriate branch ('maint' for
backward-compatible bug fixes for the last stable release, 'master' for
new features and everything else).

Please make your commits well-organized and atomic, using `git rebase
--interactive` as needed.  Open an issue or RFC pull request to discuss
any significant changes before investing time.  Check that the tests
pass using `make prove` or `make test`.  If adding a new feature, please
add or extend a test so that it is tested.

Write commit messages for a reviewer of your pull request and for a
future developer (maybe you) that bisects and finds that a bug was
introduced in your commit.  The assumptions that are clear in your head
right now are likely not.  Give credit where credit is due using tags
such as `Reported-by: Helpful User <helpful@example.com>`.  Please use a
real name and email for your author information (`git config user.name`
and `user.email`).

Please avoid "merging from upstream" (like merging 'master' into your
feature branch) unless there is a specific reason to do so, in which
case you should explain why in the merge commit.
[Rationale](https://lwn.net/Articles/328436/) from
[Junio](https://gitster.livejournal.com/42247.html) and
[Linus](http://yarchive.net/comp/linux/git_merges_from_upstream.html).

By submitting a pull request, you are affirming the following.

## [Developer's Certificate of Origin 1.1](https://developercertificate.org/).

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


