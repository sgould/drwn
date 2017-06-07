/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2017, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    drwnXMLUtils.cpp
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/

// C++ standard library
#include <cstdio>
#include <cassert>

// Eigen library
#include "Eigen/Core"

// Darwin library
#include "drwnBase.h"

using namespace std;
using namespace Eigen;

// drwnBase64 ----------------------------------------------------------------

const char *drwnBase64::LUT =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const char drwnBase64::FILL = '=';
bool drwnBase64::INSERT_NEWLINES = false;

char *drwnBase64::encode(const unsigned char *data, size_t length)
{
    // determine output length
    size_t outputLength = 4 * (length + 2) / 3 + 1;
    if (INSERT_NEWLINES) {
        outputLength += (outputLength + 79) / 80;
    }

    // allocate output buffer
    char *output = new char[outputLength];
    if (length == 0) {
        output[0] = '\0';
        return output;
    }

    DRWN_ASSERT(data != NULL);

    // convert every 3 bytes to 4 ascii characters
    char *p = &output[0];
    const unsigned char *q = &data[0];
    for ( ; length > 2; length -= 3) {
        const unsigned int buffer = (q[0] << 16) | (q[1] << 8) | q[2];
        p[0] = LUT[(buffer >> 18) & 0x3f];
        p[1] = LUT[(buffer >> 12) & 0x3f];
        p[2] = LUT[(buffer >> 6) & 0x3f];
        p[3] = LUT[buffer & 0x3f];
        p += 4; q += 3;

        // insert newline every 80 characters
        if ((INSERT_NEWLINES) && ((q - data) % 60 == 0)) {
            *p++ = '\n';
        }
    }

    // deal with non-3-byte aligned data
    switch (length) {
    case 2:
        p[0] = LUT[(q[0] >> 2) & 0x3f];
        p[1] = LUT[((q[0] << 4) | (q[1] >> 4)) & 0x3f];
        p[2] = LUT[(q[1] << 2) & 0x3f];
        p[3] = drwnBase64::FILL;
        p += 4;
        break;
    case 1:
        p[0] = LUT[(q[0] >> 2) & 0x3f];
        p[1] = LUT[(q[0] << 4) & 0x3f];
        p[2] = drwnBase64::FILL;
        p[3] = drwnBase64::FILL;
        p += 4;
        break;
    case 0:
        break;
    default:
        DRWN_ASSERT(false);
    }

    *p = '\0';
    return output;
}

unsigned char *drwnBase64::decode(const char *data)
{
    DRWN_ASSERT(data != NULL);

    size_t length = strlen(data);

    // trim trailing space
    while (isspace(data[length - 1])) {
        length -= 1;
    }

    // allocate output buffer
    size_t outlen = 3 * length / 4; // over-estimate
    if (data[length - 2] == drwnBase64::FILL) outlen -= 2;
    else if (data[length - 1] == drwnBase64::FILL) outlen -= 1;

    unsigned char *output = new unsigned char[outlen];

    // parse
    unsigned char *p = &output[0];
    const char *q = &data[0];
    for ( ; length > 0; length -= 4) {
        unsigned int buffer = 0;
        for (int i = 0; i < 4; i++) {
            buffer <<= 6;
            while (isspace(q[i])) {
                length -= 1; q++;
            }
            if ((q[i] >= 'A') && (q[i] <= 'Z'))
                buffer += (unsigned int)(q[i] - 'A');
            else if ((q[i] >= 'a') && (q[i] <= 'z'))
                buffer += (unsigned int)(q[i] - 'a' + 26);
            else if ((q[i] >= '0') && (q[i] <= '9'))
                buffer += (unsigned int)(q[i] - '0' + 52);
            else if (q[i] == '+')
                buffer += 62;
            else if (q[i] == '/')
                buffer += 63;
            else if (q[i] != drwnBase64::FILL) {
                DRWN_LOG_ERROR("base64 string: " << data << " (" << strlen(data) << ")");
                DRWN_LOG_FATAL("invalid base64 character '" << q[i] << "' at position " << (strlen(data) - length - i));
            }
        }

        p[0] = (buffer >> 16) & 0xff;
        if (q[2] != drwnBase64::FILL)
            p[1] = (buffer >> 8) & 0xff;
        if (q[3] != drwnBase64::FILL)
            p[2] = buffer & 0xff;

        p += 3; q += 4;
    }

    return output;
}

// drwnXMLUtils --------------------------------------------------------------

drwnXMLEncoderType drwnXMLUtils::DEFAULT_ENCODER = DRWN_XML_BASE64;

drwnXMLEncoderType drwnXMLUtils::getEncoderType(const char *str)
{
    if (!strcasecmp(str, "TEXT")) {
        return DRWN_XML_TEXT;
    } else if (!strcasecmp(str, "BASE64")) {
        return DRWN_XML_BASE64;
    }

    DRWN_LOG_FATAL("unrecognized encoder type " << str);
    return DRWN_XML_BASE64;
}

drwnXMLNode& drwnXMLUtils::serialize(drwnXMLNode& xml, const char *buffer, size_t length)
{
    switch (drwnXMLUtils::DEFAULT_ENCODER) {
    case DRWN_XML_TEXT:
    case DRWN_XML_BASE64:
        {
            drwnAddXMLAttribute(xml, "encoder", "base64", false);
            char *e = drwnBase64::encode((const unsigned char *)buffer, length);
            drwnAddXMLText(xml, e);
            delete[] e;
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized XML encoder");
    }

    return xml;
}

drwnXMLNode& drwnXMLUtils::serialize(drwnXMLNode& xml, const Eigen::VectorXd& v)
{
    drwnAddXMLAttribute(xml, "rows", toString(v.rows()).c_str(), false);

    switch (drwnXMLUtils::DEFAULT_ENCODER) {
    case DRWN_XML_TEXT:
        {
            drwnAddXMLAttribute(xml, "encoder", "text", false);
            std::stringstream buffer;
            buffer << v.transpose();
            drwnAddXMLText(xml, buffer.str().c_str());
        }
        break;
    case DRWN_XML_BASE64:
        {
            drwnAddXMLAttribute(xml, "encoder", "base64", false);
            char *e = drwnBase64::encode((const unsigned char *)v.data(),
                v.rows() * sizeof(double));
            drwnAddXMLText(xml, e);
            delete[] e;
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized XML encoder");
    }

    return xml;
}

drwnXMLNode& drwnXMLUtils::serialize(drwnXMLNode& xml, const Eigen::MatrixXd& m)
{
    drwnAddXMLAttribute(xml, "rows", toString(m.rows()).c_str(), false);
    drwnAddXMLAttribute(xml, "cols", toString(m.cols()).c_str(), false);

    switch (drwnXMLUtils::DEFAULT_ENCODER) {
    case DRWN_XML_TEXT:
        {
            xml.append_attribute(xml.document()->allocate_attribute("encoder", "text"));
            std::stringstream buffer;
            buffer << m;
            drwnAddXMLText(xml, buffer.str().c_str());
        }
        break;
    case DRWN_XML_BASE64:
        {
            xml.append_attribute(xml.document()->allocate_attribute("encoder", "base64"));
            char *e = drwnBase64::encode((const unsigned char *)m.data(),
                m.rows() * m.cols() * sizeof(double));
            drwnAddXMLText(xml, e);
            delete[] e;
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized XML encoder");
    }

    return xml;
}

drwnXMLNode& drwnXMLUtils::deserialize(drwnXMLNode& xml, char *buffer, size_t length)
{
    drwnXMLEncoderType encoder = DRWN_XML_BASE64;
    drwnXMLAttr *a = xml.first_attribute("encoder");
    if (a != NULL) {
        encoder = getEncoderType(a->value());
    }
    const char *data = drwnGetXMLText(xml);

    switch (encoder) {
    case DRWN_XML_TEXT:
        {
            DRWN_ASSERT(strlen(data) < length);
            strncpy(buffer, data, length);
        }
        break;
    case DRWN_XML_BASE64:
        {
            unsigned char *e = drwnBase64::decode(data);
            memcpy((void *)buffer, e, length);
            delete[] e;
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized XML encoder in " << xml.name());
    }

    return xml;
}

drwnXMLNode& drwnXMLUtils::deserialize(drwnXMLNode& xml, Eigen::VectorXd& v)
{
    drwnXMLAttr *a = xml.first_attribute("rows");
    DRWN_ASSERT(a != NULL);

    const int rows = atoi(a->value());
    if (rows == 0) {
        v = VectorXd();
        return xml;
    }
    v = VectorXd::Zero(rows);

    drwnXMLEncoderType encoder = DRWN_XML_BASE64;
    a = xml.first_attribute("encoder");
    if (a != NULL) {
        encoder = getEncoderType(a->value());
    }

    switch (encoder) {
    case DRWN_XML_TEXT:
        {
            std::stringstream buffer;
            buffer << drwnGetXMLText(xml);
            for (int r = 0; r < v.rows(); r++) {
                buffer >> v[r];
            }
        }
        break;
    case DRWN_XML_BASE64:
        {
            unsigned char *e = drwnBase64::decode(drwnGetXMLText(xml));
            memcpy((void *)v.data(), e, v.rows() * sizeof(double));
            delete[] e;
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized XML encoder in " << xml.name());
    }

    return xml;
}

drwnXMLNode& drwnXMLUtils::deserialize(drwnXMLNode& xml, Eigen::MatrixXd& m)
{
    drwnXMLAttr *a = xml.first_attribute("rows");
    DRWN_ASSERT(a != NULL);
    const int rows = atoi(a->value());

    a = xml.first_attribute("cols");
    DRWN_ASSERT(a != NULL);
    const int cols = atoi(a->value());

    if ((rows == 0) || (cols == 0)) {
        m = MatrixXd();
        return xml;
    }
    m = MatrixXd::Zero(rows, cols);

    drwnXMLEncoderType encoder = DRWN_XML_BASE64;
    a = xml.first_attribute("encoder");
    if (a != NULL) {
        encoder = getEncoderType(a->value());
    }

    switch (encoder) {
    case DRWN_XML_TEXT:
        {
            std::stringstream buffer;
            buffer << drwnGetXMLText(xml);
            for (int r = 0; r < m.rows(); r++) {
                for (int c = 0; c < m.cols(); c++) {
                    buffer >> m(r, c);
                }
            }
        }
        break;
    case DRWN_XML_BASE64:
        {
            unsigned char *e = drwnBase64::decode(drwnGetXMLText(xml));
            memcpy((void *)m.data(), e, m.rows() * m.cols() * sizeof(double));
            delete[] e;
        }
        break;
    default:
        DRWN_LOG_FATAL("unrecognized XML encoder in " << xml.name());
    }

    return xml;
}

void drwnXMLUtils::dump(drwnXMLNode& xml)
{
    rapidxml::print(cout, xml, rapidxml::print_no_indenting);
    cout << endl;
}

// drwnXMLUtilsConfig -------------------------------------------------------

//! \addtogroup drwnConfigSettings
//! \section drwnXMLUtils
//! \b encoder :: blob encoding method (TEXT, BASE64 (default))\n
//! \b prettyBase64 :: produce 80-column base64 encoding (default: false)

class drwnXMLUtilsConfig : public drwnConfigurableModule {
public:
    drwnXMLUtilsConfig() : drwnConfigurableModule("drwnXMLUtils") { }
    ~drwnXMLUtilsConfig() { }

    void usage(ostream &os) const {
        os << "      encoder       :: blob encoding method (TEXT, BASE64 (default))\n";
        os << "      prettyBase64  :: produce 80-column base64 encoding (default: false))\n";
    }

    void setConfiguration(const char *name, const char *value) {
        if (!strcmp(name, "encoder")) {
            drwnXMLUtils::DEFAULT_ENCODER = drwnXMLUtils::getEncoderType(value);
        } else if (!strcmp(name, "prettyBase64")) {
            drwnBase64::INSERT_NEWLINES = drwn::trueString(value);
        } else {
            DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
        }
    }
};

static drwnXMLUtilsConfig gXMLUtilsConfig;
