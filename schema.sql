CREATE TABLE    account{accid, accnum, ppid, pmtid};
CREATE TABLE    trigger{trigid, trigcode, pmtid};
CREATE TABLE    policyperiod{ppid, state, periodstart, periodend, policynum, pmtid};
CREATE TABLE    policydriver{poldrid, dob, pmtid};
CREATE TABLE    coverage{covid, code, value, pmtid};
CREATE TABLE    namedinsured{niid, type, pmtid};